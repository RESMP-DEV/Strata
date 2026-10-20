// src/core/kv_coordinator_test.cpp - the coordinator's state machine, epoch rules, and the store's refusal paths.
#include "strata/core/kv_coordinator.hpp"
#include "strata/core/kv_read_batch.hpp"

#include <cstdio>
#include <cstring>
#include <atomic>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

using namespace strata::core;

namespace {

KvTierCoordinator::Options small_options() {
    KvTierCoordinator::Options o;
    o.l1_groups = 2;
    o.geometry_id = 0x1111;
    o.format_id = 0x2222;
    o.blocks_per_group = 4;   // a small group, so the tests do not need 32 blocks
    o.block_bytes = 64;
    o.runs = {KvBlockRun{32}, KvBlockRun{32}};
    return o;
}

KvTierCoordinator::Options async_options() {
    KvTierCoordinator::Options o = small_options();
    o.async_io = true;
    o.async_workers = 1;
    o.l1_groups = 1;
    return o;
}

/// Append a whole group of blocks with a recognizable payload.
bool fill_group(KvTierCoordinator& c, int32_t layer, int32_t group, uint8_t value, std::string& error) {
    std::vector<uint8_t> block(64, 0);
    for (int32_t i = 0; i < 4; ++i) {
        std::memset(block.data(), value + (uint8_t) i, block.size());
        const int32_t block_index = group * 4 + i;
        if (!c.append(layer, block_index, error)) return false;
        if (!c.write_block(layer, block_index, block.data(), block.size(), error)) return false;
        if (!c.complete(layer, block_index, error)) return false;
    }
    return true;
}

/// A real store whose get path can be held at a host boundary, so cancellation is deterministic.
class BlockingStore : public KvBlockStore {
public:
    bool contains(const KvBlockKey& key) const override { return memory.contains(key); }
    bool get(const KvBlockRequest& request, std::string& error) override {
        ++gets;
        while (block_gets.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return memory.get(request, error);
    }
    bool put(const KvBlockPut& put_request, std::string& error) override { return memory.put(put_request, error); }
    bool invalidate(const KvBlockKey& key, std::string& error) override { return memory.invalidate(key, error); }
    void block_reads() { block_gets.store(true, std::memory_order_release); }
    void unblock_reads() { block_gets.store(false, std::memory_order_release); }
    size_t size() const { return memory.size(); }
    uint64_t read_count() const { return gets.load(std::memory_order_acquire); }

private:
    KvMemoryStore memory;
    std::atomic<bool> block_gets{false};
    std::atomic<uint64_t> gets{0};
};

}  // namespace

int main() {
    // 1. The L1 cache is a real cache: a group that is in L1 is a hit, one that is not is a miss that reads the
    //    store, and the bytes survive the round trip.
    {
        KvMemoryStore store;
        KvTierCoordinator c({7, 0, 0}, small_options(), &store);
        if (c.epoch() != 0) return 1;

        std::string error;
        if (!fill_group(c, 0, 0, 0x10, error)) { std::printf("fill failed: %s\n", error.c_str()); return 2; }
        if (c.completed_blocks(0, 0) != 0) return 3;   // the group was published, so it is no longer open

        // The group was published, and the slot that holds it is still in L1: a read is a hit.
        if (!c.read(0, 0, error)) { std::printf("read failed: %s\n", error.c_str()); return 4; }
        if (c.counters().l1_hits != 1) return 5;
        if (c.counters().l1_misses != 0) return 6;
        if (c.tier_of(0, 0) != KvTier::kL1) return 7;

        // The bytes are the ones that were written.
        std::vector<uint8_t> out(64, 0);
        if (!c.read_block(0, 0, out.data(), out.size(), error)) return 8;
        for (size_t i = 0; i < out.size(); ++i) {
            if (out[i] != (uint8_t) (0x10 + i / 64)) { std::printf("byte %zu is %u\n", i, out[i]); return 9; }
        }

        // A second read of the same group is a hit and does not read the store again.
        if (!c.read(0, 0, error)) return 10;
        if (c.counters().l1_hits != 2) return 11;
        if (c.counters().l2_reads != 0) return 12;
    }

    // 2. A group that is not whole is never published, so the store must not hold it.
    {
        KvMemoryStore store;
        KvTierCoordinator c({8, 0, 0}, small_options(), &store);
        std::string error;
        for (int32_t block = 0; block < 3; ++block) {
            if (!c.append(0, block, error)) return 20;
            if (!c.complete(0, block, error)) return 21;
        }
        if (c.completed_blocks(0, 0) != 3) return 22;
        if (store.writes() != 0) return 23;   // a partial group is not published
    }

    // 3. The tail is pinned: a pinned slot is never the CLOCK victim, so another group cannot evict it.
    {
        KvMemoryStore store;
        KvTierCoordinator::Options o = small_options();
        o.l1_groups = 1;
        KvTierCoordinator c({9, 0, 0}, o, &store);
        std::string error;
        if (!c.append(0, 0, error)) return 30;
        if (!c.complete(0, 0, error)) return 31;
        if (c.read(0, 1, error)) return 32;   // no slot may be taken from the pinned tail
        if (c.tier_of(0, 0) != KvTier::kL1) return 33;
    }

    // 4. A rewind advances the epoch, so the old key is unreachable and the old bytes are not served.
    {
        KvMemoryStore store;
        KvTierCoordinator c({10, 0, 0}, small_options(), &store);
        std::string error;
        if (!fill_group(c, 0, 0, 0x20, error)) return 40;
        if (!c.read(0, 0, error)) return 41;
        if (!c.rewind(error)) return 42;
        if (c.epoch() != 1) return 43;
        if (c.slot_of(0, 0) >= 0) return 44;   // the L1 cache was dropped
        if (c.read(0, 0, error)) { std::printf("a stale group was served: %s\n", error.c_str()); return 45; }
    }

    // 5. A reset invalidates the store's groups, so the old bytes cannot come back.
    {
        KvMemoryStore store;
        KvTierCoordinator c({12, 0, 0}, small_options(), &store);
        std::string error;
        if (!fill_group(c, 0, 0, 0x30, error)) return 50;
        if (store.size() != 1) return 51;
        if (!c.reset(error)) return 52;
        if (store.size() != 0) return 53;
        if (c.epoch() != 1) return 54;
    }

    // 6. A checksum mismatch is a refusal, never a zero fill. The group must be evicted first, so the read
    //    really goes to the store: that is the cold path the cold tier exists for.
    {
        KvMemoryStore store;
        KvTierCoordinator::Options o = small_options();
        o.l1_groups = 1;
        KvTierCoordinator c({13, 0, 0}, o, &store);
        std::string error;
        if (!fill_group(c, 0, 0, 0x40, error)) return 60;
        KvBlockKey key;
        key.session_id = 13;
        key.geometry_id = 0x1111;
        key.format_id = 0x2222;
        key.layer = 0;
        key.group = 0;
        key.epoch = 0;
        store.corrupt(key);
        // The second group evicts the first from L1, so the next read is a cold read of a corrupted group.
        if (!fill_group(c, 0, 1, 0x50, error)) return 63;
        if (c.slot_of(0, 0) >= 0) return 64;
        if (c.read(0, 0, error)) { std::printf("a corrupt group was served: %s\n", error.c_str()); return 61; }
        if (c.counters().checksum_errors != 1) return 62;
        if (c.counters().l2_reads != 1) return 65;   // the cold read really happened
    }

    // 7. Two coordinators with different scopes never share state, even for the same layer ordinal.
    {
        KvMemoryStore store;
        KvTierCoordinator a({14, 0, 0}, small_options(), &store);
        KvTierCoordinator b({14, 0, 1}, small_options(), &store);
        KvTierCoordinator stage1({14, 1, 0}, small_options(), &store);
        std::string error;
        if (!fill_group(a, 0, 0, 0x50, error)) return 70;
        if (!a.read(0, 0, error)) return 71;
        if (b.slot_of(0, 0) >= 0) return 72;
        if (stage1.slot_of(0, 0) >= 0) return 73;
        if (b.scope() == a.scope()) return 74;
    }

    // 8. The L1 cache evicts the least recently used unpinned slot, and the counter moves.
    {
        KvMemoryStore store;
        KvTierCoordinator::Options o = small_options();
        o.l1_groups = 2;
        KvTierCoordinator c({15, 0, 0}, o, &store);
        std::string error;
        if (!fill_group(c, 0, 0, 0x60, error)) return 80;
        if (!fill_group(c, 0, 1, 0x70, error)) return 81;
        if (c.counters().l1_evictions != 0) return 82;   // two groups fit in two slots
        if (!fill_group(c, 0, 2, 0x80, error)) return 86;
        if (c.counters().l1_evictions == 0) return 87;   // the third group evicted the first
        if (c.slot_of(0, 0) >= 0) return 88;            // the first group is gone from L1
        if (!c.read(0, 0, error)) return 83;
        if (c.counters().l1_misses != 1) return 84;
        if (c.counters().l2_reads != 1) return 85;   // the evicted group came back from the store
    }

    // 9. The store's ABI: a read past the payload, and a put with no payload, are refusals.
    {
        KvMemoryStore store;
        std::string error;
        KvBlockKey key;
        key.session_id = 16;
        key.geometry_id = 1;
        key.format_id = 2;
        key.layer = 0;
        key.group = 0;
        key.epoch = 0;
        KvBlockPut put_request;
        put_request.key = key;
        put_request.src = nullptr;
        put_request.bytes = 0;
        if (store.put(put_request, error)) return 90;
        std::vector<uint8_t> payload(16, 0);
        put_request.src = payload.data();
        put_request.bytes = payload.size();
        if (!store.put(put_request, error)) return 91;
        if (!store.contains(key)) return 92;
        std::vector<uint8_t> out(32, 0);
        KvBlockRequest request;
        request.key = key;
        request.dst = out.data();
        request.offset = 8;
        request.bytes = 32;   // past the payload
        if (store.get(request, error)) return 93;
    }

    // 10. Opt-in Phase C publication is asynchronous but explicit: the bytes are snapshotted, the store commit
    // completes off the caller's path, and the group is not consumed or unpinned until the completion is drained.
    {
        KvMemoryStore store;
        KvTierCoordinator c({17, 0, 0}, async_options(), &store);
        std::string error;
        if (!fill_group(c, 0, 0, 0x90, error)) return 100;
        std::vector<uint8_t> published(64, 0);
        if (!c.read_block(0, 0, published.data(), published.size(), error)) return 101;
        if (!c.group_io_ready(0, 0)) return 105;   // draining waits for and consumes the store completion
        if (store.writes() != 1 || c.counters().published_groups != 1) return 102;
        if (c.counters().async_writes_submitted != 1) return 103;
        if (c.slot_of(0, 0) < 0) return 104;       // the completed group remains cached and evictable
    }

    // 11. A canceled foreground read is a refusal, never a late byte installation.
    {
        BlockingStore store;
        KvTierCoordinator c({18, 0, 0}, async_options(), &store);
        std::string error;
        if (!fill_group(c, 0, 0, 0xa0, error)) return 110;
        if (!fill_group(c, 0, 1, 0xb0, error)) return 111;   // evicts group 0 from the one-slot L1
        std::vector<uint8_t> second(64, 0);
        if (!c.read_block(0, 4, second.data(), second.size(), error)) return 112;
        if (c.slot_of(0, 0) >= 0) return 112;

        store.block_reads();
        KvReadBatch batch(0, KvReadKind::kForeground);
        if (!batch.add(0, 0, -1)) return 113;
        if (!c.submit_read(batch, error)) { std::printf("submit failed: %s\n", error.c_str()); return 114; }
        const uint64_t operation = batch.items().front().operation;
        if (operation == 0 || c.group_io_ready(0, 0)) return 115;
        if (!c.cancel_read(operation, error)) return 116;
        if (c.wait_read(operation, 0, 0, error)) return 117;
        if (c.counters().async_cancellations != 1 || c.counters().l2_read_failures != 1) return 118;
        if (!c.group_io_ready(0, 0)) return 119;
        if (c.slot_of(0, 0) >= 0) return 120;

        // Release the transport and retry the same cold group through the explicit batch API.
        store.unblock_reads();
        KvReadBatch retry(0, KvReadKind::kForeground);
        if (!retry.add(0, 0, -1)) return 121;
        if (!c.submit_read(retry, error)) return 122;
        if (!c.wait_read(retry, error)) return 123;
        if (c.slot_of(0, 0) < 0 || !c.group_io_ready(0, 0)) return 124;
        std::vector<uint8_t> out(64, 0);
        if (!c.read_block(0, 0, out.data(), out.size(), error)) return 125;
        if (out.front() != 0xa0 || out.back() != 0xa0) return 126;
        if (c.counters().l2_reads != 2 || c.counters().l1_evictions == 0) return 127;
    }

    std::puts("kv coordinator: OK");
    return 0;
}

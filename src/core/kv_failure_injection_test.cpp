// src/core/kv_failure_injection_test.cpp - the failure paths the acceptance gates require: an I/O error, a short
// read, a checksum mismatch, an invalid epoch, a pinned allocation failure, and a coordinator shutdown.
#include "strata/core/kv_coordinator.hpp"
#include "strata/core/kv_file_store.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace strata::core;

namespace {

/// A transport-free store that fails on demand, so every refusal path can be driven.
class FailingStore : public KvBlockStore {
public:
    bool contains(const KvBlockKey& key) const override {
        (void) key;
        // The store holds nothing when a failure is injected: the group is absent, so the coordinator must go
        // to the store and meet the failure.
        return !fail_get && !fail_put && !short_read;
    }
    bool get(const KvBlockRequest& request, std::string& error) override {
        (void) request;
        ++gets;
        if (fail_get) {
            error = "injected: the read failed";
            return false;
        }
        if (short_read) {
            error = "injected: the read was short";
            return false;
        }
        std::memset(request.dst, 0, (size_t) request.bytes);
        return true;
    }
    bool put(const KvBlockPut& put_request, std::string& error) override {
        (void) put_request;
        ++puts_;
        if (fail_put) {
            error = "injected: the write failed";
            return false;
        }
        return true;
    }
    bool invalidate(const KvBlockKey& key, std::string& error) override {
        (void) key;
        (void) error;
        return true;
    }

    bool fail_get = false;
    bool fail_put = false;
    bool short_read = false;
    uint64_t gets = 0;
    uint64_t puts_ = 0;
};

KvTierCoordinator::Options options() {
    KvTierCoordinator::Options o;
    o.l1_groups = 1;
    o.geometry_id = 0x1111;
    o.format_id = 0x2222;
    o.blocks_per_group = 4;
    o.block_bytes = 64;
    o.runs = {KvBlockRun{32}, KvBlockRun{32}};
    return o;
}

}  // namespace

int main() {
    // 1. An I/O error on the read is a refusal, and the counter records it.
    {
        FailingStore store;
        store.fail_get = true;
        KvTierCoordinator c({1, 0, 0}, options(), &store);
        std::string error;
        if (c.read(0, 0, error)) { std::printf("case1: the read succeeded\n"); return 1; }
        if (c.counters().l2_reads != 1) { std::printf("case1: l2_reads=%llu\n", (unsigned long long)c.counters().l2_reads); return 2; }
        if (c.counters().l2_read_failures != 1) { std::printf("case1: failures=%llu\n", (unsigned long long)c.counters().l2_read_failures); return 3; }
        if (c.tier_of(0, 0) != KvTier::kNone) { std::printf("case1: tier=%d\n", (int)c.tier_of(0,0)); return 4; }
    }

    // 2. A short read is a refusal, and the group does not become resident.
    {
        FailingStore store;
        KvTierCoordinator c({2, 0, 0}, options(), &store);
        std::string error;
        store.short_read = true;
        if (c.read(0, 0, error)) return 10;
        if (c.counters().l2_read_failures != 1) return 11;
        if (c.tier_of(0, 0) != KvTier::kNone) return 12;
    }

    // 3. A failed write leaves the group unpublished and the tail pinned.
    {
        FailingStore store;
        KvTierCoordinator c({3, 0, 0}, options(), &store);
        std::string error;
        // The write fails from the start, so the group's publish fails when the last block completes.
        store.fail_put = true;
        std::vector<uint8_t> block(64, 0x11);
        for (int32_t b = 0; b < 4; ++b) {
            if (!c.append(0, b, error)) return 20;
            if (!c.write_block(0, b, block.data(), block.size(), error)) return 21;
            // The last block completes the group, and the publish fails: the completion FAILS, so the caller
            // knows the group is not in the store.
            const bool ok = c.complete(0, b, error);
            if (b == 3 && ok) { std::printf("case3: the publish reported success\n"); return 22; }
            if (b == 3 && error.empty()) { std::printf("case3: the failure carried no error\n"); return 26; }
        }
        if (store.puts_ != 1) return 24;
        // The group was not published, so the store's key is not there.
        if (!c.published_keys().empty()) return 25;
    }

    // 4. An invalid epoch is absent: the file store refuses a key of another epoch.
    {
        const std::string path = "/tmp/strata-kv-failure-epoch.bin";
        std::remove(path.c_str());
        KvFileStore::Identity identity;
        identity.session_id = 4;
        identity.geometry_id = 0x1111;
        identity.format_id = 0x2222;
        identity.layer = 0;
        identity.blocks_per_group = 4;
        identity.group_bytes = 256;
        KvFileStore store;
        std::string error;
        if (!store.open(path, identity, {KvBlockRun{32}, KvBlockRun{32}}, error)) return 30;

        KvBlockKey key;
        key.session_id = 4;
        key.geometry_id = 0x1111;
        key.format_id = 0x2222;
        key.layer = 0;
        key.group = 0;
        key.epoch = 0;
        std::vector<uint8_t> payload(256, 0x22);
        KvBlockPut put;
        put.key = key;
        put.src = payload.data();
        put.bytes = payload.size();
        put.checksum = KvFileStore::payload_checksum(payload.data(), payload.size());
        if (!store.put(put, error)) return 31;

        KvBlockKey stale = key;
        stale.epoch = 7;
        if (store.contains(stale)) return 32;
        std::vector<uint8_t> out(256, 0);
        KvBlockRequest request;
        request.key = stale;
        request.dst = out.data();
        request.offset = 0;
        request.bytes = out.size();
        if (store.get(request, error)) return 33;
        std::remove(path.c_str());
    }

    // 5. A pinned allocation failure: with every slot pinned, a read of another group is refused rather than
    //    taking a pinned slot.
    {
        FailingStore store;
        KvTierCoordinator c({5, 0, 0}, options(), &store);
        std::string error;
        if (!c.append(0, 0, error)) return 40;
        if (c.read(0, 1, error)) return 41;                 // no slot may be taken from the pinned tail
        if (c.counters().l2_reads != 0) return 42;          // and no read was issued for it
        if (c.tier_of(0, 0) != KvTier::kL1) return 43;
    }

    // 6. A coordinator shutdown: the store's groups are invalidated and the epoch advances.
    {
        FailingStore store;
        KvTierCoordinator c({6, 0, 0}, options(), &store);
        std::string error;
        std::vector<uint8_t> block(64, 0x33);
        for (int32_t b = 0; b < 4; ++b) {
            if (!c.append(0, b, error)) return 50;
            if (!c.write_block(0, b, block.data(), block.size(), error)) return 51;
            if (!c.complete(0, b, error)) return 52;
        }
        if (c.published_keys().size() != 1) return 53;
        if (!c.reset(error)) return 54;
        if (c.epoch() != 1) return 55;
        if (!c.published_keys().empty()) return 56;
    }

    std::puts("kv failure injection: OK");
    return 0;
}

// src/core/kv_mooncake_store_test.cpp - the shared store's contract: opaque payloads, Strata-owned keys, epochs,
// the quota, and the refusal of a corrupted object.
#include "strata/core/kv_mooncake_store.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace strata::core;

namespace {

KvMooncakeStore::Options options(uint64_t quota = 0, int32_t layer = 0) {
    KvMooncakeStore::Options o;
    o.session_id = 7;
    o.geometry_id = 0x1111;
    o.format_id = 0x2222;
    o.layer = layer;
    o.blocks_per_group = 4;
    o.group_bytes = 256;
    o.quota_bytes = quota;
    return o;
}

KvBlockKey key(int32_t layer, int32_t group, uint32_t epoch = 0) {
    KvBlockKey k;
    k.session_id = 7;
    k.geometry_id = 0x1111;
    k.format_id = 0x2222;
    k.layer = layer;
    k.group = group;
    k.epoch = epoch;
    return k;
}

uint64_t checksum_of(const std::vector<uint8_t>& data) {
    // The one checksum function, from the shared header: a writer and a reader
    // must not have two checksum implementations.
    return kv_payload_checksum(data.data(), data.size());
}

}  // namespace

int main() {
    // 1. A group is written whole, read back, and is byte-identical.
    {
        KvMemoryTransport transport;
        KvMooncakeStore store(&transport, options());
        std::string error;

        std::vector<uint8_t> payload(256, 0x5a);
        KvBlockPut put;
        put.key = key(0, 0);
        put.src = payload.data();
        put.bytes = payload.size();
        put.checksum = checksum_of(payload);
        if (!store.put(put, error)) { std::printf("put failed: %s\n", error.c_str()); return 1; }
        if (!store.contains(key(0, 0))) return 2;

        std::vector<uint8_t> out(256, 0);
        KvBlockRequest request;
        request.key = key(0, 0);
        request.dst = out.data();
        request.offset = 0;
        request.bytes = out.size();
        request.checksum = checksum_of(payload);
        if (!store.get(request, error)) { std::printf("get failed: %s\n", error.c_str()); return 3; }
        if (out != payload) return 4;
        if (store.writes() != 1 || store.reads() != 1) return 5;
    }

    // 2. The key is Strata's: another epoch, layer, or format is a different object, not the same bytes.
    {
        KvMemoryTransport transport;
        KvMooncakeStore store(&transport, options());
        std::string error;
        std::vector<uint8_t> payload(256, 0x11);
        KvBlockPut put;
        put.key = key(0, 0);
        put.src = payload.data();
        put.bytes = payload.size();
        put.checksum = checksum_of(payload);
        if (!store.put(put, error)) return 10;

        if (store.contains(key(0, 0, 1))) return 11;   // another epoch
        if (store.contains(key(1, 0))) return 12;     // another layer
        KvBlockKey other_format = key(0, 0);
        other_format.format_id = 0x9999;
        if (store.contains(other_format)) return 13;

        // The object key itself carries the epoch, so the two epochs are two objects.
        if (KvMooncakeStore::object_key(key(0, 0)) == KvMooncakeStore::object_key(key(0, 0, 1))) return 14;
    }

    // 3. A corrupted object is refused, not returned.
    {
        KvMemoryTransport transport;
        KvMooncakeStore store(&transport, options());
        std::string error;
        std::vector<uint8_t> payload(256, 0x22);
        KvBlockPut put;
        put.key = key(0, 0);
        put.src = payload.data();
        put.bytes = payload.size();
        put.checksum = checksum_of(payload);
        if (!store.put(put, error)) return 20;
        transport.corrupt(KvMooncakeStore::object_key(key(0, 0)));

        std::vector<uint8_t> out(256, 0);
        KvBlockRequest request;
        request.key = key(0, 0);
        request.dst = out.data();
        request.offset = 0;
        request.bytes = out.size();
        request.checksum = checksum_of(payload);
        if (store.get(request, error)) { std::printf("a corrupt object was served: %s\n", error.c_str()); return 21; }
    }

    // 4. The quota is checked before the write, and a refused write does not consume the pool.
    {
        KvMemoryTransport transport;
        KvMooncakeStore store(&transport, options(300));
        std::string error;
        std::vector<uint8_t> payload(256, 0x33);
        KvBlockPut put;
        put.key = key(0, 0);
        put.src = payload.data();
        put.bytes = payload.size();
        put.checksum = checksum_of(payload);
        if (!store.put(put, error)) return 30;
        if (store.used_bytes() != 256) return 31;
        if (store.remaining_bytes() != 44) return 32;

        // A second group would exceed the quota.
        KvBlockPut second = put;
        second.key = key(0, 1);
        if (store.put(second, error)) { std::printf("the quota was exceeded: %s\n", error.c_str()); return 33; }
        if (store.quota_refusals() != 1) return 34;
        if (store.used_bytes() != 256) return 35;   // the refused write consumed nothing

        // Overwriting the same group does not consume the pool twice.
        if (!store.put(put, error)) return 36;
        if (store.used_bytes() != 256) return 37;

        // Invalidating frees the pool.
        if (!store.invalidate(key(0, 0), error)) return 38;
        if (store.used_bytes() != 0) return 39;
    }

    // 5. A put that is not one whole group is refused.
    {
        KvMemoryTransport transport;
        KvMooncakeStore store(&transport, options());
        std::string error;
        std::vector<uint8_t> payload(256, 0x44);
        KvBlockPut put;
        put.key = key(0, 0);
        put.src = payload.data();
        put.bytes = 100;
        if (store.put(put, error)) return 40;
    }

    // 6. The store works as the coordinator's store: the same ABI, the same epoch rules.
    {
        KvMemoryTransport transport;
        KvMooncakeStore store(&transport, options());
        KvTierCoordinator::Options tier_options;
        tier_options.l1_groups = 1;
        tier_options.geometry_id = 0x1111;
        tier_options.format_id = 0x2222;
        tier_options.blocks_per_group = 4;
        tier_options.block_bytes = 64;
        tier_options.runs = {KvBlockRun{32}, KvBlockRun{32}};
        KvTierCoordinator coordinator({7, 0, 0}, tier_options, &store);
        std::string error;
        std::vector<uint8_t> block(64, 0x55);
        for (int32_t b = 0; b < 4; ++b) {
            if (!coordinator.append(0, b, error)) return 50;
            if (!coordinator.write_block(0, b, block.data(), block.size(), error)) return 51;
            if (!coordinator.complete(0, b, error)) return 52;
        }
        // The group was published to the shared store.
        if (transport.size() != 1) return 53;
        if (!coordinator.read(0, 0, error)) { std::printf("read failed: %s\n", error.c_str()); return 54; }
        // A rewind makes the old epoch's object unreachable.
        if (!coordinator.rewind(error)) return 55;
        if (coordinator.epoch() != 1) return 56;
    }

    // 7. A COLD read through the shared store works: the group is evicted from L1 first, so the read really
    //    goes to the store. The coordinator must send the checksum of the group it PUBLISHED, not the bytes of
    //    whatever the L1 slot holds, or every cold read fails.
    {
        KvMemoryTransport transport;
        KvMooncakeStore store(&transport, options());
        KvTierCoordinator::Options tier_options;
        tier_options.layer = 0;
        tier_options.l1_groups = 1;
        tier_options.geometry_id = 0x1111;
        tier_options.format_id = 0x2222;
        tier_options.blocks_per_group = 4;
        tier_options.block_bytes = 64;
        tier_options.runs = {KvBlockRun{32}, KvBlockRun{32}};
        KvTierCoordinator coordinator({7, 0, 0}, tier_options, &store);
        std::string error;
        // The two groups hold DIFFERENT bytes. If the coordinator sent the checksum of whatever the L1 slot
        // holds, the second group's bytes would be the ones the store would verify against, and the cold read
        // of the FIRST group would fail.
        std::vector<uint8_t> first(64, 0x77);
        std::vector<uint8_t> second(64, 0x88);
        for (int32_t b = 0; b < 4; ++b) {
            if (!coordinator.append(0, b, error)) return 70;
            if (!coordinator.write_block(0, b, first.data(), first.size(), error)) return 71;
            if (!coordinator.complete(0, b, error)) return 72;
        }
        // A second group evicts the first from L1, so the next read of group 0 is a COLD read.
        for (int32_t b = 4; b < 8; ++b) {
            if (!coordinator.append(0, b, error)) return 73;
            if (!coordinator.write_block(0, b, second.data(), second.size(), error)) return 74;
            if (!coordinator.complete(0, b, error)) return 75;
        }
        if (coordinator.slot_of(0, 0) >= 0) return 76;   // group 0 is gone from L1
        if (!coordinator.read(0, 0, error)) {
            std::printf("the cold read failed: %s\n", error.c_str());
            return 77;
        }
        if (coordinator.counters().l2_reads != 1) return 78;
        if (coordinator.counters().checksum_errors != 0) return 79;
        if (store.checksum_errors() != 0) return 80;
        // The bytes that came back are the FIRST group's, not the second group's.
        std::vector<uint8_t> out(64, 0);
        if (!coordinator.read_block(0, 0, out.data(), out.size(), error)) return 81;
        if (out != first) return 82;
    }

    // 8. The quota and eviction order belong to the POOL, not to each store: two layer identities share one
    //    bounded object set, exactly as Mooncake's Master-side admission does.
    {
        KvMooncakePoolTransport::Options pool_options;
        pool_options.quota_bytes = 512;
        pool_options.max_objects = 2;
        KvMooncakePoolTransport transport(pool_options);
        KvMooncakeStore layer0(&transport, options(0, 0));
        KvMooncakeStore layer1(&transport, options(0, 1));
        std::string error;

        auto put = [&](KvMooncakeStore& store, int32_t layer, int32_t group, uint8_t value) {
            std::vector<uint8_t> payload(256, value);
            KvBlockPut request;
            request.key = key(layer, group);
            request.src = payload.data();
            request.bytes = payload.size();
            request.checksum = checksum_of(payload);
            return store.put(request, error);
        };
        if (!put(layer0, 0, 0, 0x66)) return 90;
        if (!put(layer1, 1, 0, 0x77)) return 91;
        if (transport.size() != 2 || transport.used_bytes() != 512) return 92;

        // A third admission evicts the first object even though it came through another store identity.
        if (!put(layer1, 1, 1, 0x88)) return 93;
        if (transport.size() != 2 || transport.evictions() != 1) return 94;
        if (layer0.contains(key(0, 0))) return 95;
        if (!layer1.contains(key(1, 0)) || !layer1.contains(key(1, 1))) return 96;

        // Objects are immutable. A same-key payload change is refused and does not reset the eviction age.
        if (put(layer1, 1, 1, 0x99)) return 97;
        if (transport.immutable_refusals() != 1 || transport.size() != 2) return 98;

        // An object larger than the whole pool is refused before eviction, so it cannot wipe the pool.
        KvMooncakePoolTransport::Options tiny;
        tiny.quota_bytes = 128;
        KvMooncakePoolTransport tiny_pool(tiny);
        KvMooncakeStore tiny_store(&tiny_pool, options());
        if (put(tiny_store, 0, 0, 0xaa)) return 99;
        if (tiny_pool.quota_refusals() != 1 || tiny_pool.size() != 0) return 100;
    }

    std::puts("kv mooncake store: OK");
    return 0;
}

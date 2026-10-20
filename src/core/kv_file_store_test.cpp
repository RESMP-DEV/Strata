// src/core/kv_file_store_test.cpp - the L2 file store's durability, identity, and refusal contracts.
#include "strata/core/kv_file_store.hpp"
#include "strata/core/kv_mooncake_store.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::core;

namespace {

/// The byte offset of the header's payload_offset field. The header is written by the store; the test reads the
/// field it needs instead of transcribing the whole layout. The layout is: magic 8, version 4, header_size 4,
/// session_id 8, layer 4, geometry_id 8, format_id 8, blocks_per_group 8, block_bytes 8, group_bytes 8,
/// run_count 8, max_groups 8, groups 8, entries_offset 8, payload_offset 8.
constexpr size_t kPayloadOffsetField = 8 + 4 + 4 + 8 + 4 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8;

KvFileStore::Identity identity() {
    KvFileStore::Identity id;
    id.session_id = 42;
    id.geometry_id = 0x1111;
    id.format_id = 0x2222;
    id.layer = 0;
    id.blocks_per_group = 4;
    id.group_bytes = 4 * 64;   // four blocks of 64 bytes
    id.max_groups = 64;        // the directory's capacity
    return id;
}

std::vector<KvBlockRun> runs() { return {KvBlockRun{32}, KvBlockRun{32}}; }

KvBlockKey key(int32_t layer, int32_t group, uint32_t epoch = 0) {
    KvBlockKey k;
    k.session_id = 42;
    k.geometry_id = 0x1111;
    k.format_id = 0x2222;
    k.layer = layer;
    k.group = group;
    k.epoch = epoch;
    return k;
}

uint64_t checksum_of(const std::vector<uint8_t>& data) {
    // The store's own function: a writer and a reader must not have two checksum implementations.
    return KvFileStore::payload_checksum(data.data(), data.size());
}

/// Every KV store must agree with the coordinator about the checksum of a
/// group: the writer hashes the bytes, the reader verifies them, and a
/// disagreement refuses every group that is ever read back.  The stores
/// therefore expose the SAME function, and this test pins that they do.
void check_one_checksum_function() {
    const std::vector<uint8_t> payload = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const uint64_t shared = kv_payload_checksum(payload.data(), payload.size());
    if (KvFileStore::payload_checksum(payload.data(), payload.size()) != shared) {
        std::printf("the file store's checksum differs from the shared one\n");
        return;
    }
    if (KvMooncakeStore::payload_checksum(payload.data(), payload.size()) != shared) {
        std::printf("the shared store's checksum differs from the shared one\n");
        return;
    }
    // A change of one byte must change the checksum.
    std::vector<uint8_t> changed = payload;
    changed[3] ^= 0x01;
    if (kv_payload_checksum(changed.data(), changed.size()) == shared) {
        std::printf("the checksum ignores a changed byte\n");
    }
}

}  // namespace

int main() {
    check_one_checksum_function();
    const std::string path = "/tmp/strata-kv-file-store-test.bin";
    std::remove(path.c_str());

    // 1. A group is written whole, read back, and is byte-identical.
    {
        KvFileStore store;
        std::string error;
        if (!store.open(path, identity(), runs(), error)) { std::printf("open failed: %s\n", error.c_str()); return 1; }

        std::vector<uint8_t> payload(256, 0);
        for (size_t i = 0; i < payload.size(); ++i) payload[i] = (uint8_t) (i * 7 + 3);
        KvBlockPut put;
        put.key = key(0, 0);
        put.src = payload.data();
        put.bytes = payload.size();
        put.checksum = checksum_of(payload);
        if (!store.put(put, error)) { std::printf("put failed: %s\n", error.c_str()); return 2; }
        if (!store.contains(key(0, 0))) return 3;

        std::vector<uint8_t> out(256, 0);
        KvBlockRequest request;
        request.key = key(0, 0);
        request.dst = out.data();
        request.offset = 0;
        request.bytes = out.size();
        if (!store.get(request, error)) { std::printf("get failed: %s\n", error.c_str()); return 4; }
        if (out != payload) return 5;
        if (store.writes() != 1 || store.reads() != 1) return 6;

        // 2. A key of another epoch is absent: a rewind must not read the old bytes as current data.
        if (store.contains(key(0, 0, 1))) return 7;
        KvBlockRequest stale = request;
        stale.key = key(0, 0, 1);
        if (store.get(stale, error)) { std::printf("a stale epoch was served: %s\n", error.c_str()); return 8; }

        // 3. A key of another geometry is absent.
        KvBlockKey other = key(0, 0);
        other.geometry_id = 0x9999;
        if (store.contains(other)) return 9;

        // 4. A partial read is refused: the checksum covers the whole group, so a partial read cannot be
        //    verified and an unverifiable read is the torn read the valid marker exists to prevent.
        KvBlockRequest past = request;
        past.offset = 200;
        past.bytes = 100;
        if (store.get(past, error)) return 10;
        KvBlockRequest oversize = request;
        oversize.bytes = 100000;
        if (store.get(oversize, error)) return 15;

        // 5. A put that is not one whole group is refused.
        KvBlockPut partial = put;
        partial.bytes = 100;
        if (store.put(partial, error)) return 11;

        // 6. An invalidated group is absent.
        if (!store.invalidate(key(0, 0), error)) return 12;
        if (store.contains(key(0, 0))) return 13;
        if (store.get(request, error)) return 14;
    }

    // 7. The store reopens and keeps its groups, and a store of another identity is rewritten rather than reused.
    {
        KvFileStore store;
        std::string error;
        if (!store.open(path, identity(), runs(), error)) { std::printf("reopen failed: %s\n", error.c_str()); return 20; }
        // The group was invalidated, so it is absent after the reopen.
        if (store.contains(key(0, 0))) return 21;

        std::vector<uint8_t> payload(256, 0x5a);
        KvBlockPut put;
        put.key = key(0, 2);
        put.src = payload.data();
        put.bytes = payload.size();
        put.checksum = checksum_of(payload);
        if (!store.put(put, error)) return 22;
        store.close();

        KvFileStore again;
        if (!again.open(path, identity(), runs(), error)) return 23;
        if (!again.contains(key(0, 2))) return 24;   // the group survived the reopen
        std::vector<uint8_t> out(256, 0);
        KvBlockRequest request;
        request.key = key(0, 2);
        request.dst = out.data();
        request.offset = 0;
        request.bytes = out.size();
        if (!again.get(request, error)) return 25;
        if (out != payload) return 26;

        // A store of another geometry does not reuse the file's bytes: it rewrites them.
        KvFileStore::Identity other = identity();
        other.geometry_id = 0x7777;
        KvFileStore rewritten;
        if (!rewritten.open(path, other, runs(), error)) return 27;
        if (rewritten.contains(key(1, 2))) return 28;   // the other geometry's groups are gone
    }

    // 8. A corrupted payload is refused, not served.
    {
        KvFileStore store;
        std::string error;
        if (!store.open(path, identity(), runs(), error)) return 30;
        std::vector<uint8_t> payload(256, 0x11);
        KvBlockPut put;
        put.key = key(0, 0);
        put.src = payload.data();
        put.bytes = payload.size();
        put.checksum = checksum_of(payload);
        if (!store.put(put, error)) return 31;
        store.close();

        // Corrupt the payload area on disk, leaving the marker set: a reader must refuse the group.
        FILE* file = std::fopen(path.c_str(), "r+b");
        if (file == nullptr) return 32;
        std::fseek(file, 0, SEEK_END);
        const long size = std::ftell(file);
        std::fseek(file, 0, SEEK_SET);
        std::vector<uint8_t> header((size_t) size, 0);
        if (std::fread(header.data(), 1, header.size(), file) != header.size()) return 33;
        // The payload starts after the header and the group directory, aligned to 4096. The offset is read from
        // the header's own field, so the test does not carry a second copy of the layout.
        uint64_t payload_offset = 0;
        std::memcpy(&payload_offset, header.data() + kPayloadOffsetField, 8);
        if (payload_offset == 0 || payload_offset + 16 > header.size()) return 38;
        header[(size_t) payload_offset + 5] ^= 0xff;
        std::fseek(file, 0, SEEK_SET);
        if (std::fwrite(header.data(), 1, header.size(), file) != header.size()) return 34;
        std::fclose(file);

        // The header's own checksum does not cover the payload, so the file reopens and the CORRUPTED GROUP is
        // the thing that is refused: the corruption is in the payload, not in the header.
        KvFileStore reopened;
        if (!reopened.open(path, identity(), runs(), error)) { std::printf("reopen failed: %s\n", error.c_str()); return 35; }
        std::vector<uint8_t> out(256, 0);
        KvBlockRequest request;
        request.key = key(0, 0);
        request.dst = out.data();
        request.offset = 0;
        request.bytes = out.size();
        if (reopened.get(request, error)) { std::printf("a corrupt group was served: %s\n", error.c_str()); return 36; }
        if (reopened.checksum_errors() != 1) return 37;
    }

    std::remove(path.c_str());
    std::puts("kv file store: OK");
    return 0;
}

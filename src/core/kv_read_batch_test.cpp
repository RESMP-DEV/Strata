// src/core/kv_read_batch_test.cpp - the read batch's deduplication and layering rules.
#include "strata/core/kv_read_batch.hpp"

#include <cstdio>

using namespace strata::core;

int main() {
    // 1. The same group named twice appears once, and the work the deduplication saved is counted.
    {
        KvReadBatch batch(0, KvReadKind::kForeground);
        if (!batch.add(0, 5, 3)) return 1;
        if (!batch.add(0, 5, 3)) return 2;
        if (!batch.add(0, 6, 4)) return 3;
        if (batch.size() != 2) return 4;
        if (batch.requested() != 3) return 5;
        if (batch.items()[0].group != 5 || batch.items()[0].slot != 3) return 6;
        if (batch.items()[1].group != 6 || batch.items()[1].slot != 4) return 7;
    }

    // 2. A duplicate with no slot does not move a group that already has one.
    {
        KvReadBatch batch(0, KvReadKind::kForeground);
        batch.add(0, 5, 3);
        batch.add(0, 5, -1);
        if (batch.items()[0].slot != 3) return 10;
        // A duplicate with a slot fills a slot that was unknown.
        KvReadBatch other(0, KvReadKind::kForeground);
        other.add(0, 5, -1);
        other.add(0, 5, 7);
        if (other.items()[0].slot != 7) return 11;
    }

    // 3. A batch never mixes layers.
    {
        KvReadBatch batch(0, KvReadKind::kForeground);
        batch.add(0, 1, 0);
        if (batch.add(1, 1, 0)) return 20;
        if (batch.size() != 1) return 21;
        if (batch.layer() != 0) return 22;
    }

    // 4. The kind carries the priority, and the deadline is what a foreground batch must not miss.
    {
        KvReadBatch batch(0, KvReadKind::kPrefetch);
        if (batch.kind() != KvReadKind::kPrefetch) return 30;
        batch.set_kind(KvReadKind::kForeground);
        if (batch.kind() != KvReadKind::kForeground) return 31;
        batch.set_deadline_us(5000);
        if (batch.deadline_us() != 5000) return 32;
        batch.set_bytes(4096);
        if (batch.bytes() != 4096) return 33;
    }

    // 5. An empty batch is empty.
    {
        KvReadBatch batch(0, KvReadKind::kForeground);
        if (!batch.empty()) return 40;
    }

    std::puts("kv read batch: OK");
    return 0;
}

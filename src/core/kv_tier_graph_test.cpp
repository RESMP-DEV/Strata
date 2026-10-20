// src/core/kv_tier_graph_test.cpp - the graph-split proof: the captured sequence must split at the selection
// boundary, because a CUDA graph cannot perform host-side file I/O.
#include "strata/core/kv_tier_session.hpp"

#include <cstdio>

using namespace strata::core;

int main() {
    // A window that goes selection -> readiness -> resolve is a proper split.
    KvTierSession::note_phase(kKvTierPhaseSelection);
    KvTierSession::note_phase(kKvTierPhaseReadiness);
    KvTierSession::note_phase(kKvTierPhaseResolve);
    const KvTierGraphPhases one = KvTierSession::graph_phases();
    if (one.selection != 1 || one.readiness != 1 || one.resolve != 1) return 1;
    if (one.windows != 1) return 2;
    if (one.split_violations != 0) return 3;

    // A resolve with no host readiness is a violation: the graph tried to do host work.
    KvTierSession::note_phase(kKvTierPhaseResolve);
    const KvTierGraphPhases two = KvTierSession::graph_phases();
    if (two.split_violations != 1) return 4;
    if (two.windows != 1) return 5;   // the bad window did not count as a complete window

    // A readiness with no selection is a violation too.
    KvTierSession::note_phase(kKvTierPhaseReadiness);
    const KvTierGraphPhases three = KvTierSession::graph_phases();
    if (three.split_violations != 2) return 6;

    // Two clean windows in a row.
    KvTierSession::note_phase(kKvTierPhaseSelection);
    KvTierSession::note_phase(kKvTierPhaseReadiness);
    KvTierSession::note_phase(kKvTierPhaseResolve);
    KvTierSession::note_phase(kKvTierPhaseSelection);
    KvTierSession::note_phase(kKvTierPhaseReadiness);
    KvTierSession::note_phase(kKvTierPhaseResolve);
    const KvTierGraphPhases four = KvTierSession::graph_phases();
    if (four.windows != 3) return 7;
    if (four.split_violations != 2) return 8;
    // The counters count every phase, including the phases of the violating windows.
    if (four.selection != 3) return 9;
    if (four.readiness != 4) return 10;   // three clean plus the one that had no selection
    if (four.resolve != 4) return 11;     // three clean plus the one that had no readiness

    std::puts("kv tier graph split: OK");
    return 0;
}

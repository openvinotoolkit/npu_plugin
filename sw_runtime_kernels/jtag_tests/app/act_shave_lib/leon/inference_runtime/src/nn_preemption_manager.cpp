/*
 * {% copyright %}
 */
#include "nn_preemption_manager.h"
#include <mv_types.h>
#include <nn_log.h>

namespace nn {
namespace inference_runtime {
namespace preemption {

using namespace common_runtime;

void sendPreemptionToNNShaves(uint8_t tile) {
    UNUSED(tile);
}

void sendPreemptionToACTShaves(uint8_t tile) {
    UNUSED(tile);
}

void preemptDmaLinkAgent(uint8_t engine, uint8_t agent) {
    UNUSED(engine);
    UNUSED(agent);
}

bool waitForNNShavesPreemption(uint8_t tile) {
    UNUSED(tile);
    return false;
}

bool waitForACTShavesPreemption(uint8_t tile) {
    UNUSED(tile);
    return false;
}

bool waitForDmaLinkAgent(uint8_t engine, uint8_t agent) {
    UNUSED(engine);
    UNUSED(agent);
    return false;
}

bool recordNNShavesPreemptionState(uint8_t tile, PreemptionState *state) {
    UNUSED(tile);
    UNUSED(state);
    return false;
}

bool recordACTShavesPreemptionState(uint8_t tile, PreemptionState *state) {
    UNUSED(tile);
    UNUSED(state);
    return false;
}

bool recordDmaLinkAgentState(uint8_t engine, uint8_t agent, PreemptionState *state) {
    UNUSED(engine);
    UNUSED(agent);
    UNUSED(state);
    return false;
}

} // namespace preemption
} // namespace inference_runtime
} // namespace nn

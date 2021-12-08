/*
 * {% copyright %}
 */
#pragma once

#include <mv_types.h>
#include <nn_hw_resources.h>

using namespace nn::common_runtime;

namespace nn {
namespace inference_runtime {
namespace preemption {

struct SNNPreemptionState {};
struct ASPreemptionState {};
struct DMAPreemptionState {};
struct PreemptionState {
    SNNPreemptionState snnState;
    ASPreemptionState asState;
    DMAPreemptionState dmaState;
};

void sendPreemptionToNNShaves(uint8_t tile);
void sendPreemptionToACTShaves(uint8_t tile);
void preemptDmaLinkAgent(uint8_t engine, uint8_t agent);

bool waitForNNShavesPreemption(uint8_t tile);
bool waitForACTShavesPreemption(uint8_t tile);
bool waitForDmaLinkAgent(uint8_t engine, uint8_t agent);

/// a preemption must have been sent and waited to use record*State()
bool recordNNShavesPreemptionState(uint8_t tile, PreemptionState *state);
/// a preemption must have been sent and waited to use record*State()
bool recordACTShavesPreemptionState(uint8_t tile, PreemptionState *state);
/// a preemption must have been sent and waited to use record*State()
bool recordDmaLinkAgentState(uint8_t engine, uint8_t agent, PreemptionState *state);

class PreemptionManager {};

} // namespace preemption
} // namespace inference_runtime
} // namespace nn

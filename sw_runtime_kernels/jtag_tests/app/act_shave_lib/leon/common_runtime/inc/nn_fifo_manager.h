/*
 * {% copyright %}
 */
#pragma once

#include <mv_types.h>
#include <nn_fifo_configs.h>
#include <nn_ctrl_manager.h>

namespace nn {
namespace common_runtime {
namespace fifo {

struct SHVFifoConfig {
    FifoConfig work; /// used for WorkItem retreval
    FifoConfig ctrx; /// used for ctrl messages: exit, preempt, etc.
    FifoConfig cttx; /// used to send ack/response to ctrl messages
    FifoConfig perf; /// used for always-on perf data streaming / debug stream out
};

struct SNNCtrlMessage {
    SHVCtrlMessage message;
};

static_assert(sizeof(SNNCtrlMessage) < 4, "SNNCtrlMessage should fit into a 32b word");

struct SNNCtrlResponse {};

struct SNNPerfReport {};

struct ASCtrlMessage {
    SHVCtrlMessage message;
};

static_assert(sizeof(ASCtrlMessage) < 4, "ASCtrlMessage should fit into a 32b word");

struct ASCtrlResponse {};

struct ASPerfReport {};

constexpr uint32_t packSHVConfig(SHVFifoConfig cfg) {
    uint32_t ret{0};

    ret |= ((uint32_t)cfg.work.fifo & 0b1111) << (32 - 4 * 1);
    ret |= ((uint32_t)cfg.work.index & 0b1111) << (32 - 4 * 2);
    ret |= ((uint32_t)cfg.ctrx.fifo & 0b1111) << (32 - 4 * 3);
    ret |= ((uint32_t)cfg.ctrx.index & 0b1111) << (32 - 4 * 4);
    ret |= ((uint32_t)cfg.cttx.fifo & 0b1111) << (32 - 4 * 5);
    ret |= ((uint32_t)cfg.cttx.index & 0b1111) << (32 - 4 * 6);
    ret |= ((uint32_t)cfg.perf.fifo & 0b1111) << (32 - 4 * 7);
    ret |= ((uint32_t)cfg.perf.index & 0b1111) << (32 - 4 * 8);

    return ret;
}

// ShaveNN configs
constexpr uint32_t snn0_t0_cfg{packSHVConfig({snnWorkFifo[0][0], snnCtrlRxFifo[0][0],
                                            snnCtrlTxFifo[0][0], snnPerfFifo[0][0]})};
constexpr uint32_t snn0_t1_cfg{packSHVConfig({snnWorkFifo[1][0], snnCtrlRxFifo[1][0],
                                            snnCtrlTxFifo[1][0], snnPerfFifo[1][0]})};
constexpr uint32_t snn_cfgs[SNN_PER_TILE * MAX_TILES] = {snn0_t0_cfg, snn0_t1_cfg};

// ACTSHV configs
constexpr uint32_t acts0_t0_cfg{packSHVConfig({actWorkFifo[0][0], actCtrlRxFifo[0][0],
                                            actCtrlTxFifo[0][0], actPerfFifo[0][0]})};
constexpr uint32_t acts1_t0_cfg{packSHVConfig({actWorkFifo[0][1], actCtrlRxFifo[0][1],
                                            actCtrlTxFifo[0][1], actPerfFifo[0][1]})};
constexpr uint32_t acts0_t1_cfg{packSHVConfig({actWorkFifo[1][0], actCtrlRxFifo[1][0],
                                            actCtrlTxFifo[1][0], actPerfFifo[1][0]})};
constexpr uint32_t acts1_t1_cfg{packSHVConfig({actWorkFifo[1][1], actCtrlRxFifo[1][1],
                                            actCtrlTxFifo[1][1], actPerfFifo[1][1]})};
constexpr uint32_t acts_cfgs[AS_PER_TILE * MAX_TILES] = {acts0_t0_cfg, acts1_t0_cfg, acts0_t1_cfg, acts1_t1_cfg};

SHVFifoConfig unpackSHVConfig(uint32_t packedCfg);

#if defined(__leon__) || defined(__leon_nn__)

uint32_t packSNNCtrlMessage(SNNCtrlMessage cm);
uint32_t packASCtrlMessage(ASCtrlMessage cm);

SNNCtrlResponse unpackSNNCtrlResponse(uint32_t cr);
ASCtrlResponse unpackASCtrlResponse(uint32_t cr);

SNNPerfReport unpackSNNPerfReport(uint64_t pr);
ASPerfReport unpackASPerfReport(uint64_t pr);

bool isSNNWorkFifoFull(uint8_t tile);
bool isASWorkFifoFull(uint8_t tile);

bool isSNNCtrlFifoFull(uint8_t tile, uint8_t snn);
bool isASCtrlFifoFull(uint8_t tile, uint8_t as);

/// work queue shared for all SNNs per tile
void sendWorkToSNNs(uint8_t tile, void *p);
/// work queue shared for all ActSHVs per tile
void sendWorkToASs(uint8_t tile, void *p);

void sendCtrlToSNNs(uint8_t tile, uint8_t snn, void *p);
void sendCtrlToASs(uint8_t tile, uint8_t as, void *p);

void printFifoConfig (SHVFifoConfig config);

// FIXME: we could further limit compiled code per shave type if there was a `defined(__act_shave__)`
#elif defined(__shave__) || defined(__shave_nn__) || defined(__act_shave__)

uint32_t packSNNCtrlResponse(SNNCtrlResponse cr);
uint32_t packASCtrlResponse(ASCtrlResponse cr);

uint64_t packSNNPerfReport(SNNPerfReport pr);
uint64_t packASPerfReport(ASPerfReport pr);

SNNCtrlMessage unpackSNNCtrlMessage(uint32_t cm);
ASCtrlMessage unpackASCtrlMessage(uint32_t cm);

#endif

} // namespace fifo
} // namespace common_runtime
} // namespace nn

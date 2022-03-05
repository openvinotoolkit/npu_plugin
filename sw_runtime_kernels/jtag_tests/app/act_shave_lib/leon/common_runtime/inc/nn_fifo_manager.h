//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <mv_types.h>
#include <nn_fifo_configs.h>
#include <limits.h>

namespace nn {
namespace common_runtime {
namespace fifo {

#define MAX_CTRL_MESSAGE_BITS 4
#define MAX_CTRL_PAYLOAD_BITS (32 - MAX_CTRL_MESSAGE_BITS)
#define CTRL_MESSAGE_MASK ((UINT32_MAX >> MAX_CTRL_PAYLOAD_BITS) << MAX_CTRL_PAYLOAD_BITS)

enum SHVCtrlMessage : uint8_t {
    NoOp = 0,
    HWStatsEnable,
    PreemptHaltAndAck,
    EnablePerfStream,
    DisablePerfStream,
    Shutdown // we don't need this -- a NULL work item already does this
};

// We don't _need_ the Ctrl message to be a 32b word, but it makes things easier
static_assert(SHVCtrlMessage::Shutdown < (1 << MAX_CTRL_MESSAGE_BITS),
              "SHVCtrlMessage needs to be compressable to 4 bits");

struct SHVFifoConfig {
    FifoConfig work; /// used for WorkItem retreval
    FifoConfig ctrx; /// used for ctrl messages: exit, preempt, etc.
    FifoConfig cttx; /// used to send ack/response to ctrl messages
    FifoConfig perf; /// used for always-on perf data streaming / debug stream out
};

struct SNNCtrlMessage {
    SNNCtrlMessage(SHVCtrlMessage message, uint32_t payload) {
        this->message = message;
        this->payload = payload;
    }

    SHVCtrlMessage message;
    uint32_t payload;
};
//struct SNNCtrlMessage {
//    SHVCtrlMessage message;
//    uint32_t payload;
//};

struct SNNCtrlResponse {};

struct ASCtrlMessage {
    ASCtrlMessage(SHVCtrlMessage message, uint32_t payload) {
        this->message = message;
        this->payload = payload;
    }

    SHVCtrlMessage message;
    uint32_t payload;
};

struct ASCtrlResponse {};

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
constexpr uint32_t snn0_t0_cfg{
    packSHVConfig({snnWorkFifo[0][0], snnCtrlRxFifo[0][0], snnCtrlTxFifo[0][0], snnPerfFifo[0][0]})};
constexpr uint32_t snn0_t1_cfg{
    packSHVConfig({snnWorkFifo[1][0], snnCtrlRxFifo[1][0], snnCtrlTxFifo[1][0], snnPerfFifo[1][0]})};
constexpr uint32_t snn_cfgs[SNN_PER_TILE * MAX_TILES] = {snn0_t0_cfg, snn0_t1_cfg};

// ACTSHV configs
constexpr uint32_t acts0_t0_cfg{
    packSHVConfig({actWorkFifo[0][0], actCtrlRxFifo[0][0], actCtrlTxFifo[0][0], actPerfFifo[0][0]})};
constexpr uint32_t acts1_t0_cfg{
    packSHVConfig({actWorkFifo[0][1], actCtrlRxFifo[0][1], actCtrlTxFifo[0][1], actPerfFifo[0][1]})};
constexpr uint32_t acts0_t1_cfg{
    packSHVConfig({actWorkFifo[1][0], actCtrlRxFifo[1][0], actCtrlTxFifo[1][0], actPerfFifo[1][0]})};
constexpr uint32_t acts1_t1_cfg{
    packSHVConfig({actWorkFifo[1][1], actCtrlRxFifo[1][1], actCtrlTxFifo[1][1], actPerfFifo[1][1]})};
constexpr uint32_t acts_cfgs[AS_PER_TILE * MAX_TILES] = {acts0_t0_cfg, acts1_t0_cfg, acts0_t1_cfg, acts1_t1_cfg};

SHVFifoConfig unpackSHVConfig(uint32_t packedCfg);

#if defined(__leon__) || defined(__leon_nn__)

uint32_t packSNNCtrlMessage(SNNCtrlMessage cm);
uint32_t packASCtrlMessage(ASCtrlMessage cm);

SNNCtrlResponse unpackSNNCtrlResponse(uint32_t cr);
ASCtrlResponse unpackASCtrlResponse(uint32_t cr);

bool isSNNWorkFifoFull(uint8_t tile);
bool isASWorkFifoFull(uint8_t tile);

bool isSNNCtrlFifoFull(uint8_t tile, uint8_t snn);
bool isASCtrlFifoFull(uint8_t tile, uint8_t as);

/// work queue shared for all SNNs per tile
void sendWorkToSNNs(uint8_t tile, void *p);
/// work queue shared for all ActSHVs per tile
void sendWorkToASs(uint8_t tile, void *p);

// we go though all this packing/unpacking trouble for ctrl messages so that we don't have to solve:
// 1) making sure the message sent is in user context accesible space like *p in sendWorkTo*()
// 2) avoid all the fun concurancy issues we may have if the message de-scoped before a shave reads it
// 3) if *p were moved to DDR one day, there would be cache coherency stuff to solve
void sendCtrlToSNNs(uint8_t tile, uint8_t snn, uint32_t packedMessage);
void sendCtrlToASs(uint8_t tile, uint8_t as, uint32_t packedMessage);

void printFifoConfig(SHVFifoConfig config);

// FIXME: we could further limit compiled code per shave type if there was a `defined(__act_shave__)`
#elif defined(__shave__) || defined(__shave_nn__) || defined(__act_shave__)

uint32_t packSNNCtrlResponse(SNNCtrlResponse cr);
uint32_t packASCtrlResponse(ASCtrlResponse cr);

SNNCtrlMessage unpackSNNCtrlMessage(uint32_t cm);
ASCtrlMessage unpackASCtrlMessage(uint32_t cm);

#endif

} // namespace fifo
} // namespace common_runtime
} // namespace nn

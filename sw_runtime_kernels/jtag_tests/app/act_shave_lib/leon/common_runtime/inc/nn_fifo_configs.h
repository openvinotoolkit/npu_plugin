//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

/**
 * The perf FIFO assignments are effectively halving the work FIFOs. It may be better to use FIFO2's many unused 16x64
 * FIFOs (doc says both 16x16 and 16x64)
 *
 */
#pragma once

#include <mv_types.h>
#include <nn_hw_resources.h>

namespace nn {
namespace common_runtime {
namespace fifo {

namespace { // this anonymous namespace forces everyone to use the constexpr get* accessors
// constexpr uint8_t LNN_WORK_FIFO_NUM = 2; // Will managemen tasks feed lnn via a FIFO? If so, use this.
// constexpr uint8_t LNN_WORK_FIFO_INDEX = 0;

constexpr uint8_t CTRL_FIFO = 2;

// Tile 0 Fifo assignments
constexpr uint8_t TILE0_SNN0_WORK_FIFO_NUM = 0;
constexpr uint8_t TILE0_SNN0_WORK_FIFO_INDEX = 0;
constexpr uint8_t TILE0_AS0_WORK_FIFO_NUM = 3;
constexpr uint8_t TILE0_AS0_WORK_FIFO_INDEX = 0;

constexpr uint8_t TILE0_SNN0_CTRL_RX_FIFO_NUM = CTRL_FIFO;
constexpr uint8_t TILE0_SNN0_CTRL_RX_FIFO_INDEX = 2;
constexpr uint8_t TILE0_AS0_CTRL_RX_FIFO_NUM = CTRL_FIFO;
constexpr uint8_t TILE0_AS0_CTRL_RX_FIFO_INDEX =
    SNN_PER_TILE == 1 ? TILE0_SNN0_CTRL_RX_FIFO_INDEX + 1 : TILE0_SNN0_CTRL_RX_FIFO_INDEX + 2;

constexpr uint8_t TILE0_SNN0_PERF_FIFO_NUM = 0;
constexpr uint8_t TILE0_SNN0_PERF_FIFO_INDEX = 1;
constexpr uint8_t TILE0_AS0_PERF_FIFO_NUM = 3;
constexpr uint8_t TILE0_AS0_PERF_FIFO_INDEX = 2;

// Tile 0 Fifo assignments
constexpr uint8_t TILE1_SNN0_WORK_FIFO_NUM = 1;
constexpr uint8_t TILE1_SNN0_WORK_FIFO_INDEX = 0;
constexpr uint8_t TILE1_AS0_WORK_FIFO_NUM = 3;
constexpr uint8_t TILE1_AS0_WORK_FIFO_INDEX = 1;

constexpr uint8_t TILE1_SNN0_CTRL_RX_FIFO_NUM = CTRL_FIFO;
constexpr uint8_t TILE1_SNN0_CTRL_RX_FIFO_INDEX = TILE0_AS0_CTRL_RX_FIFO_INDEX + 2;
constexpr uint8_t TILE1_AS0_CTRL_RX_FIFO_NUM = CTRL_FIFO;
constexpr uint8_t TILE1_AS0_CTRL_RX_FIFO_INDEX =
    SNN_PER_TILE == 1 ? TILE1_SNN0_CTRL_RX_FIFO_INDEX + 1 : TILE1_SNN0_CTRL_RX_FIFO_INDEX + 2;

constexpr uint8_t TILE0_SNN0_CTRL_TX_FIFO_NUM = CTRL_FIFO;
constexpr uint8_t TILE0_SNN0_CTRL_TX_FIFO_INDEX = TILE1_AS0_CTRL_RX_FIFO_INDEX + 2;
constexpr uint8_t TILE0_AS0_CTRL_TX_FIFO_NUM = CTRL_FIFO;
constexpr uint8_t TILE0_AS0_CTRL_TX_FIFO_INDEX =
    SNN_PER_TILE == 1 ? TILE0_SNN0_CTRL_TX_FIFO_INDEX + 1 : TILE0_SNN0_CTRL_TX_FIFO_INDEX + 2;

constexpr uint8_t TILE1_SNN0_CTRL_TX_FIFO_NUM = CTRL_FIFO;
constexpr uint8_t TILE1_SNN0_CTRL_TX_FIFO_INDEX = TILE0_AS0_CTRL_TX_FIFO_INDEX + 2;
constexpr uint8_t TILE1_AS0_CTRL_TX_FIFO_NUM = CTRL_FIFO;
constexpr uint8_t TILE1_AS0_CTRL_TX_FIFO_INDEX =
    SNN_PER_TILE == 1 ? TILE1_SNN0_CTRL_TX_FIFO_INDEX + 1 : TILE1_SNN0_CTRL_TX_FIFO_INDEX + 2;

constexpr uint8_t TILE1_SNN0_PERF_FIFO_NUM = 1;
constexpr uint8_t TILE1_SNN0_PERF_FIFO_INDEX = 1;
constexpr uint8_t TILE1_AS0_PERF_FIFO_NUM = 3;
constexpr uint8_t TILE1_AS0_PERF_FIFO_INDEX = 3;

} // namespace

constexpr uint32_t FIFO_TOTAL_ELEMENTS = 1024;
constexpr uint32_t FIFO0_LENGTH = 512;
constexpr uint32_t FIFO1_LENGTH = 512;
constexpr uint32_t FIFO3_LENGTH = 256;
constexpr uint32_t FIFO0_PARTS = FIFO_TOTAL_ELEMENTS / FIFO0_LENGTH;
constexpr uint32_t FIFO1_PARTS = FIFO_TOTAL_ELEMENTS / FIFO1_LENGTH;
constexpr uint32_t FIFO3_PARTS = FIFO_TOTAL_ELEMENTS / FIFO3_LENGTH;

constexpr uint16_t SNN_WORK_FIFO_NUM[MAX_TILES] = {TILE0_SNN0_WORK_FIFO_NUM, TILE1_SNN0_WORK_FIFO_NUM};
constexpr uint16_t AS_WORK_FIFO_NUM[MAX_TILES] = {TILE0_AS0_WORK_FIFO_NUM, TILE1_AS0_WORK_FIFO_NUM};
constexpr uint16_t CTRL_FIFO_NUM[MAX_TILES] = {CTRL_FIFO, CTRL_FIFO};

// validate configs
static_assert(SNN_PER_TILE == 1 || (SNN_PER_TILE > 1 && FIFO0_LENGTH < FIFO_TOTAL_ELEMENTS),
              "Fifo0's dimensions are not configured for 2 SNNs per tile");
static_assert(AS_PER_TILE > 1 && FIFO0_LENGTH < 1024, "Fifo3's dimensions are not configured for 2 ActSHVs per tile");
static_assert(TILE1_AS0_CTRL_TX_FIFO_INDEX < 15, "Overprovisioning of Fifo");

static_assert(CTRL_FIFO == TILE0_SNN0_CTRL_RX_FIFO_NUM && CTRL_FIFO == TILE0_AS0_CTRL_RX_FIFO_NUM &&
                  CTRL_FIFO == TILE1_SNN0_CTRL_RX_FIFO_NUM && CTRL_FIFO == TILE1_AS0_CTRL_RX_FIFO_NUM &&
                  CTRL_FIFO == TILE0_SNN0_CTRL_TX_FIFO_NUM && CTRL_FIFO == TILE0_AS0_CTRL_TX_FIFO_NUM &&
                  CTRL_FIFO == TILE1_SNN0_CTRL_TX_FIFO_NUM && CTRL_FIFO == TILE1_AS0_CTRL_TX_FIFO_NUM,
              "All saves should use FIFO2 for IPC control messages");

struct FifoConfig {
    uint8_t fifo;
    uint8_t index;
};

namespace {

constexpr FifoConfig localFifoSelect(bool sharedFifo, uint32_t tile, uint32_t tileLocalShvId, uint8_t fIndexT0,
                                     uint8_t fIndexT1, uint8_t fifoNumT0, uint8_t fifoNumT1) {
    // use these asserts if we ever switch to C++20 and can use consteval for these functions
    // static_assert(tileLocalShvId < SNN_PER_TILE, "Maximum 2 SNNs tiles supported");
    // static_assert(tile < MAX_TILES, "Exceeded maximum NCE tiles supported");
    if (tile == 0) {
        uint8_t index = fIndexT0 + (sharedFifo ? 0 : tileLocalShvId);
        return {fifoNumT0, index};
    } else {
        uint8_t index = fIndexT1 + (sharedFifo ? 0 : tileLocalShvId);
        return {fifoNumT1, index};
    }
}

/// NN Shaves of the same tile share the same work fifo
constexpr FifoConfig getSNNWorkFifo(uint32_t tile, uint32_t tileLocalSnnId) {
    return localFifoSelect(true, tile, tileLocalSnnId, TILE0_SNN0_WORK_FIFO_INDEX, TILE1_SNN0_WORK_FIFO_INDEX,
                           TILE0_SNN0_WORK_FIFO_NUM, TILE1_SNN0_WORK_FIFO_NUM);
}

constexpr FifoConfig getSNNCtrlRxFifo(uint32_t tile, uint32_t tileLocalSnnId) {
    return localFifoSelect(false, tile, tileLocalSnnId, TILE0_SNN0_CTRL_RX_FIFO_INDEX, TILE1_SNN0_CTRL_RX_FIFO_INDEX,
                           TILE0_SNN0_CTRL_RX_FIFO_NUM, TILE1_SNN0_CTRL_RX_FIFO_NUM);
}

constexpr FifoConfig getSNNCtrlTxFifo(uint32_t tile, uint32_t tileLocalSnnId) {
    return localFifoSelect(true, tile, tileLocalSnnId, TILE0_SNN0_CTRL_TX_FIFO_INDEX, TILE1_SNN0_CTRL_TX_FIFO_INDEX,
                           TILE0_SNN0_CTRL_TX_FIFO_NUM, TILE1_SNN0_CTRL_TX_FIFO_NUM);
}

constexpr FifoConfig getSNNPerfFifo(uint32_t tile, uint32_t tileLocalSnnId) {
    return localFifoSelect(true, tile, tileLocalSnnId, TILE0_SNN0_PERF_FIFO_INDEX, TILE1_SNN0_PERF_FIFO_INDEX,
                           TILE0_SNN0_PERF_FIFO_NUM, TILE1_SNN0_PERF_FIFO_NUM);
}
/// Activation Shaves of the same tile share the same work fifo
constexpr FifoConfig getActWorkFifo(uint32_t tile, uint32_t tileLocalActId) {
    return localFifoSelect(true, tile, tileLocalActId, TILE0_AS0_WORK_FIFO_INDEX, TILE1_AS0_WORK_FIFO_INDEX,
                           TILE0_AS0_WORK_FIFO_NUM, TILE1_AS0_WORK_FIFO_NUM);
}

constexpr FifoConfig getActCtrlRxFifo(uint32_t tile, uint32_t tileLocalActId) {
    return localFifoSelect(false, tile, tileLocalActId, TILE0_AS0_CTRL_RX_FIFO_INDEX, TILE1_AS0_CTRL_RX_FIFO_INDEX,
                           TILE0_AS0_CTRL_RX_FIFO_NUM, TILE1_AS0_CTRL_RX_FIFO_NUM);
}

constexpr FifoConfig getActCtrlTxFifo(uint32_t tile, uint32_t tileLocalActId) {
    return localFifoSelect(true, tile, tileLocalActId, TILE0_AS0_CTRL_TX_FIFO_INDEX, TILE1_AS0_CTRL_TX_FIFO_INDEX,
                           TILE0_AS0_CTRL_TX_FIFO_NUM, TILE1_AS0_CTRL_TX_FIFO_NUM);
}

constexpr FifoConfig getActPerfFifo(uint32_t tile, uint32_t tileLocalActId) {
    return localFifoSelect(true, tile, tileLocalActId, TILE0_AS0_PERF_FIFO_INDEX, TILE1_AS0_PERF_FIFO_INDEX,
                           TILE0_AS0_PERF_FIFO_NUM, TILE1_AS0_PERF_FIFO_NUM);
}
} // namespace

constexpr FifoConfig actWorkFifo[MAX_TILES][AS_PER_TILE] = {{getActWorkFifo(0, 0), getActWorkFifo(0, 1)},
                                                            {getActWorkFifo(1, 0), getActWorkFifo(1, 1)}};
constexpr FifoConfig actPerfFifo[MAX_TILES][AS_PER_TILE] = {{getActPerfFifo(0, 0), getActPerfFifo(0, 1)},
                                                            {getActPerfFifo(1, 0), getActPerfFifo(1, 1)}};
constexpr FifoConfig actCtrlRxFifo[MAX_TILES][AS_PER_TILE] = {{getActCtrlRxFifo(0, 0), getActCtrlRxFifo(0, 1)},
                                                              {getActCtrlRxFifo(1, 0), getActCtrlRxFifo(1, 1)}};
constexpr FifoConfig actCtrlTxFifo[MAX_TILES][AS_PER_TILE] = {{getActCtrlTxFifo(0, 0), getActCtrlTxFifo(0, 1)},
                                                              {getActCtrlTxFifo(1, 0), getActCtrlTxFifo(1, 1)}};

constexpr FifoConfig snnWorkFifo[MAX_TILES][SNN_PER_TILE] = {{getSNNWorkFifo(0, 0)}, {getSNNWorkFifo(1, 0)}};
constexpr FifoConfig snnPerfFifo[MAX_TILES][SNN_PER_TILE] = {{getSNNPerfFifo(0, 0)}, {getSNNPerfFifo(1, 0)}};
constexpr FifoConfig snnCtrlRxFifo[MAX_TILES][SNN_PER_TILE] = {{getSNNCtrlRxFifo(0, 0)}, {getSNNCtrlRxFifo(1, 0)}};
constexpr FifoConfig snnCtrlTxFifo[MAX_TILES][SNN_PER_TILE] = {{getSNNCtrlTxFifo(0, 0)}, {getSNNCtrlTxFifo(1, 0)}};

constexpr uint16_t snnWorkFifoIndexMask[MAX_TILES] = {1 << snnWorkFifo[0][0].index, 1 << snnWorkFifo[1][0].index};
constexpr uint16_t asWorkFifoIndexMask[MAX_TILES] = {1 << actWorkFifo[0][0].index | 1 << actWorkFifo[0][1].index,
                                                     1 << actWorkFifo[1][0].index | 1 << actWorkFifo[1][1].index};
constexpr uint16_t asPerfFifoIndexMask[MAX_TILES] = {1 << actPerfFifo[0][0].index | 1 << actPerfFifo[0][1].index,
                                                     1 << actPerfFifo[1][0].index | 1 << actPerfFifo[1][1].index};
constexpr uint16_t ctrlFifoIndexMask[MAX_TILES] = {
    1 << snnCtrlRxFifo[0][0].index | 1 << snnCtrlTxFifo[0][0].index | 1 << actCtrlRxFifo[0][0].index |
        1 << actCtrlTxFifo[0][0].index | 1 << actCtrlRxFifo[0][1].index | 1 << actCtrlTxFifo[0][1].index,
    1 << snnCtrlRxFifo[1][0].index | 1 << snnCtrlTxFifo[1][0].index | 1 << actCtrlRxFifo[1][0].index |
        1 << actCtrlTxFifo[1][0].index | 1 << actCtrlRxFifo[1][1].index | 1 << actCtrlTxFifo[1][1].index};

// constexpr FifoConfig getLNNWorkFifo(uint8_t tile, uint8_t act_shave) {}
// constexpr FifoConfig getLNNTXFifo(uint8_t tile, uint8_t act_shave) {}
// constexpr FifoConfig getLNNCtrlRxFifo(uint8_t tile, uint8_t act_shave) {}
// constexpr FifoConfig getLNNPerfFifo(uint8_t tile, uint8_t act_shave) {}

// validate that config accessors follow the rules
static_assert(getSNNWorkFifo(0, 0).fifo == TILE0_SNN0_WORK_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getSNNWorkFifo(0, 0).index == TILE0_SNN0_WORK_FIFO_INDEX, "got an unexpected FIFO index");

static_assert(getSNNCtrlRxFifo(0, 0).fifo == TILE0_SNN0_CTRL_RX_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getSNNCtrlRxFifo(0, 0).index == TILE0_SNN0_CTRL_RX_FIFO_INDEX, "got an unexpected FIFO index");
static_assert(getSNNCtrlTxFifo(0, 0).fifo == TILE0_SNN0_CTRL_TX_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getSNNCtrlTxFifo(0, 0).index == TILE0_SNN0_CTRL_TX_FIFO_INDEX, "got an unexpected FIFO index");
static_assert(getSNNPerfFifo(0, 0).fifo == TILE0_SNN0_PERF_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getSNNPerfFifo(0, 0).index == TILE0_SNN0_PERF_FIFO_INDEX, "got an unexpected FIFO index");

static_assert(getSNNWorkFifo(1, 0).fifo == TILE1_SNN0_WORK_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getSNNWorkFifo(1, 0).index == TILE1_SNN0_WORK_FIFO_INDEX, "got an unexpected FIFO index");

static_assert(getSNNCtrlRxFifo(1, 0).fifo == TILE1_SNN0_CTRL_RX_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getSNNCtrlRxFifo(1, 0).index == TILE1_SNN0_CTRL_RX_FIFO_INDEX, "got an unexpected FIFO index");
static_assert(getSNNCtrlTxFifo(1, 0).fifo == TILE1_SNN0_CTRL_TX_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getSNNCtrlTxFifo(1, 0).index == TILE1_SNN0_CTRL_TX_FIFO_INDEX, "got an unexpected FIFO index");
static_assert(getSNNPerfFifo(1, 0).fifo == TILE1_SNN0_PERF_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getSNNPerfFifo(1, 0).index == TILE1_SNN0_PERF_FIFO_INDEX, "got an unexpected FIFO index");

static_assert(getActWorkFifo(0, 0).fifo == TILE0_AS0_WORK_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActWorkFifo(0, 0).index == TILE0_AS0_WORK_FIFO_INDEX, "got an unexpected FIFO index");
static_assert(getActWorkFifo(0, 1).fifo == TILE0_AS0_WORK_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActWorkFifo(0, 1).index == TILE0_AS0_WORK_FIFO_INDEX, "got an unexpected FIFO index");

static_assert(getActCtrlRxFifo(0, 0).fifo == TILE0_AS0_CTRL_RX_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActCtrlRxFifo(0, 0).index == TILE0_AS0_CTRL_RX_FIFO_INDEX, "got an unexpected FIFO index");
static_assert(getActCtrlRxFifo(0, 1).fifo == TILE0_AS0_CTRL_RX_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActCtrlRxFifo(0, 1).index == TILE0_AS0_CTRL_RX_FIFO_INDEX + 1, "got an unexpected FIFO index");
// ActSHVs of the same tile share the same Ctrl TX
static_assert(getActCtrlTxFifo(0, 0).fifo == TILE0_AS0_CTRL_TX_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActCtrlTxFifo(0, 0).index == TILE0_AS0_CTRL_TX_FIFO_INDEX, "got an unexpected FIFO index");
static_assert(getActCtrlTxFifo(0, 1).fifo == TILE0_AS0_CTRL_TX_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActCtrlTxFifo(0, 1).index == TILE0_AS0_CTRL_TX_FIFO_INDEX, "got an unexpected FIFO index");
// ActSHVs of the same tile share the same Perf stream
static_assert(getActPerfFifo(0, 0).fifo == TILE0_AS0_PERF_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActPerfFifo(0, 0).index == TILE0_AS0_PERF_FIFO_INDEX, "got an unexpected FIFO index");
static_assert(getActPerfFifo(0, 1).fifo == TILE0_AS0_PERF_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActPerfFifo(0, 1).index == TILE0_AS0_PERF_FIFO_INDEX, "got an unexpected FIFO index");

static_assert(getActWorkFifo(1, 0).fifo == TILE1_AS0_WORK_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActWorkFifo(1, 0).index == TILE1_AS0_WORK_FIFO_INDEX, "got an unexpected FIFO index");
static_assert(getActWorkFifo(1, 1).fifo == TILE1_AS0_WORK_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActWorkFifo(1, 1).index == TILE1_AS0_WORK_FIFO_INDEX, "got an unexpected FIFO index");

static_assert(getActCtrlRxFifo(1, 0).fifo == TILE1_AS0_CTRL_RX_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActCtrlRxFifo(1, 0).index == TILE1_AS0_CTRL_RX_FIFO_INDEX, "got an unexpected FIFO index");
static_assert(getActCtrlRxFifo(1, 1).fifo == TILE1_AS0_CTRL_RX_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActCtrlRxFifo(1, 1).index == TILE1_AS0_CTRL_RX_FIFO_INDEX + 1, "got an unexpected FIFO index");
// ActSHVs of the same tile share the same Ctrl TX
static_assert(getActCtrlTxFifo(1, 0).fifo == TILE1_AS0_CTRL_TX_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActCtrlTxFifo(1, 0).index == TILE1_AS0_CTRL_TX_FIFO_INDEX, "got an unexpected FIFO index");
static_assert(getActCtrlTxFifo(1, 1).fifo == TILE1_AS0_CTRL_TX_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActCtrlTxFifo(1, 1).index == TILE1_AS0_CTRL_TX_FIFO_INDEX, "got an unexpected FIFO index");
// ActSHVs of the same tile share the same Perf stream
static_assert(getActPerfFifo(1, 0).fifo == TILE1_AS0_PERF_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActPerfFifo(1, 0).index == TILE1_AS0_PERF_FIFO_INDEX, "got an unexpected FIFO index");
static_assert(getActPerfFifo(1, 1).fifo == TILE1_AS0_PERF_FIFO_NUM, "got an unexpected FIFO number");
static_assert(getActPerfFifo(1, 1).index == TILE1_AS0_PERF_FIFO_INDEX, "got an unexpected FIFO index");

} // namespace fifo
} // namespace common_runtime
} // namespace nn

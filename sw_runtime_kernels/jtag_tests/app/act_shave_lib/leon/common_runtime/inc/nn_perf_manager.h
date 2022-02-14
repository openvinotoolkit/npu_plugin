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

namespace nn {
namespace common_runtime {
namespace perf {

enum PCCBitOffsetsBits : uint8_t {
    BIT_STALL_CYCLE_CNT_EN = 0,
    BIT_EXEC_INST_CNT_EN = 1,
    BIT_CLK_CYCLE_CNT_EN = 2,
    BIT_BRANCH_TAKEN_CNT_EN = 3,
    BIT_INST_BRKP0_CNT_EN = 4,
    BIT_INST_BRKP1_CNT_EN = 5,
    BIT_DATA_BRKP0_CNT_EN = 6,
    BIT_DATA_BRKP1_CNT_EN = 7,
    BIT_GO_COUNT_EN = 8,
    BIT_LSU0_RBYTE_CNT_EN = 9,
    BIT_LSU0_WBYTE_CNT_EN = 10,
    BIT_LSU1_RBYTE_CNT_EN = 11,
    BIT_LSU1_WBYTE_CNT_EN = 12
};

enum MetricMaskFeatureOffsetBits : uint8_t {
    STALL_CYCLE_CNT_EN = BIT_STALL_CYCLE_CNT_EN,
    EXEC_INST_CNT_EN = BIT_EXEC_INST_CNT_EN,
    CLK_CYCLE_CNT_EN = BIT_CLK_CYCLE_CNT_EN,
    BRANCH_TAKEN_CNT_EN = BIT_BRANCH_TAKEN_CNT_EN,
    INST_BRKP0_CNT_EN = BIT_INST_BRKP0_CNT_EN,
    INST_BRKP1_CNT_EN = BIT_INST_BRKP1_CNT_EN,
    DATA_BRKP0_CNT_EN = BIT_DATA_BRKP0_CNT_EN,
    DATA_BRKP1_CNT_EN = BIT_DATA_BRKP1_CNT_EN,
    GO_COUNT_EN = BIT_GO_COUNT_EN,
    LSU0_RBYTE_CNT_EN = BIT_LSU0_RBYTE_CNT_EN,
    LSU0_WBYTE_CNT_EN = BIT_LSU0_WBYTE_CNT_EN,
    LSU1_RBYTE_CNT_EN = BIT_LSU1_RBYTE_CNT_EN,
    LSU1_WBYTE_CNT_EN = BIT_LSU1_WBYTE_CNT_EN,
    FRC_DURATION_EN = 27,
    FRC_TIMESTAMP_EN = 28
};

struct SNNPerfReport {};

struct ActPerfReport {
    uint64_t timestamp;
    uint32_t duration;
    uint32_t pc0;
    uint32_t pc1;
    uint32_t pc2;
    uint32_t pc3;
};

uint32_t actPRPackedSize(uint32_t metricMask);

#if defined(__leon__) || defined(__leon_nn__)

void unpackSNNPerfReport(uint32_t metricMask, const void *prPtr, SNNPerfReport &pr);
void unpackActPerfReport(uint32_t metricMask, const void *prPtr, ActPerfReport &pr);

uint32_t buildMetricMask(bool enableTimestamp, bool enableDuration, const uint8_t *counters, uint32_t ctSize,
                         const uint8_t *stallFilters, uint32_t sfSize);

// FIXME: we could further limit compiled code per shave type if there was a `defined(__act_shave__)`
// Update: PR#8868 should fix this with "__shave__" meaning actSHV. Revisit this when the dust settles
#elif defined(__shave__) || defined(__shave_nn__) || defined(__act_shave__)

void packSNNPerfReport(uint32_t metricMask, const SNNPerfReport &pr, void *prPtr);
void packActPerfReport(uint32_t metricMask, const ActPerfReport &pr, void *prPtr);

#endif

} // namespace perf
} // namespace common_runtime
} // namespace nn

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "nn_perf_manager.h"
#include "nn_log.h"

namespace nn {
namespace common_runtime {
namespace perf {

uint32_t countCounters(uint32_t metricMask) {
    // [28:27] are the FRC values
    // [26:0]  are a direct map to the SVU_PCCx registers (with multiple perf bits allowed!)
    uint32_t i = metricMask & 0b1111111111111111111111111111; // 28 bits
    i = i - ((i >> 1) & 0x55555555);                          // add pairs of bits
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);           // quads
    i = (i + (i >> 4)) & 0x0F0F0F0F;                          // groups of 8
    return (i * 0x01010101) >> 24;                            // horizontal sum of bytes
}

uint32_t actPRPackedSize(uint32_t metricMask) {
    uint32_t size{countCounters(metricMask) * sizeof(ActPerfReport::pc0)};
    size += (metricMask & (0b1 << 28)) ? sizeof(ActPerfReport::timestamp) : 0;
    size += (metricMask & (0b1 << 27)) ? sizeof(ActPerfReport::duration) : 0;

    return size;
}

#if defined(__leon__) || defined(__leon_nn__)

void unpackSNNPerfReport(const uint32_t metricMask, const void *prPtr, SNNPerfReport &pr) {
    UNUSED(metricMask);
    UNUSED(prPtr);
    UNUSED(pr);
}

void unpackActPerfReport(const uint32_t metricMask, const void *prPtr, ActPerfReport &pr) {
    const uint32_t counters{countCounters(metricMask)};
    uint32_t byteStep{0};

    if (metricMask & (0b1 << FRC_TIMESTAMP_EN)) {
        pr.timestamp = *reinterpret_cast<const uint32_t *>(reinterpret_cast<const uint32_t>(prPtr));
        byteStep += sizeof(ActPerfReport::timestamp);
    }
    if (metricMask & (0b1 << FRC_DURATION_EN)) {
        pr.duration = *reinterpret_cast<const uint32_t *>(reinterpret_cast<const uint32_t>(prPtr) + byteStep);
        byteStep += sizeof(ActPerfReport::duration);
    }

    if (counters > 0) {
        pr.pc0 = *reinterpret_cast<const uint32_t *>(reinterpret_cast<const uint32_t>(prPtr) + byteStep);
        byteStep += sizeof(ActPerfReport::pc0);
    }
    if (counters > 1) {
        pr.pc1 = *reinterpret_cast<const uint32_t *>(reinterpret_cast<const uint32_t>(prPtr) + byteStep);
        byteStep += sizeof(ActPerfReport::pc1);
    }
    if (counters > 2) {
        pr.pc2 = *reinterpret_cast<const uint32_t *>(reinterpret_cast<const uint32_t>(prPtr) + byteStep);
        byteStep += sizeof(ActPerfReport::pc2);
    }
    if (counters > 3) {
        pr.pc3 = *reinterpret_cast<const uint32_t *>(reinterpret_cast<const uint32_t>(prPtr) + byteStep);
    }
}

uint32_t buildMetricMask(bool enableTimestamp, bool enableDuration, const uint8_t *counters, uint32_t ctSize,
                         const uint8_t *stallFilters, uint32_t sfSize) {
    uint32_t mask{0};

    if (enableTimestamp) {
        mask |= (0b1 << FRC_TIMESTAMP_EN);
    }

    if (enableDuration) {
        mask |= (0b1 << FRC_DURATION_EN);
    }

    for (uint32_t i{0}; i < ctSize; i++) {
        if (counters[i] <= BIT_LSU1_WBYTE_CNT_EN)
            mask |= (0b1 << counters[i]);
        else
            nnLog(MVLOG_ERROR,
                  "Incorrect coding of ActSHV perf counter flags from schedule: expected < %" PRId32 ", got %" PRId32
                  "",
                  BIT_LSU1_WBYTE_CNT_EN, counters[i]);
    }

    for (uint32_t i{0}; i < sfSize; i++) {
        if (stallFilters[i] < FRC_DURATION_EN)
            mask |= (0b1 << stallFilters[i]);
        else
            nnLog(MVLOG_ERROR,
                  "Incorrect coding of ActSHV perf stall filter flags from schedule: expected < %" PRId32
                  ", got %" PRId32 "",
                  FRC_DURATION_EN, stallFilters[i]);
    }

    return mask;
}

// FIXME: we could further limit compiled code per shave type if there was a `defined(__act_shave__)`
#elif defined(__shave__) || defined(__shave_nn__) || defined(__act_shave__)

void packSNNPerfReport(const uint32_t metricMask, const SNNPerfReport &pr, void *prPtr) {
    UNUSED(metricMask);
    UNUSED(pr);
    UNUSED(prPtr);
}

void packActPerfReport(const uint32_t metricMask, const ActPerfReport &pr, void *prPtr) {
    const uint32_t counters{countCounters(metricMask)};
    uint32_t byteStep{0};

    if (metricMask & (0b1 << FRC_TIMESTAMP_EN)) {
        *reinterpret_cast<uint32_t *>(prPtr) = pr.timestamp;
        byteStep += sizeof(ActPerfReport::timestamp);
    }
    if (metricMask & (0b1 << FRC_DURATION_EN)) {
        *reinterpret_cast<uint32_t *>(reinterpret_cast<uint32_t>(prPtr) + byteStep) = pr.duration;
        byteStep += sizeof(ActPerfReport::duration);
    }

    if (counters > 0) {
        *reinterpret_cast<uint32_t *>(reinterpret_cast<uint32_t>(prPtr) + byteStep) = pr.pc0;
        byteStep += sizeof(ActPerfReport::pc0);
    }
    if (counters > 1) {
        *reinterpret_cast<uint32_t *>(reinterpret_cast<uint32_t>(prPtr) + byteStep) = pr.pc1;
        byteStep += sizeof(ActPerfReport::pc1);
    }
    if (counters > 2) {
        *reinterpret_cast<uint32_t *>(reinterpret_cast<uint32_t>(prPtr) + byteStep) = pr.pc2;
        byteStep += sizeof(ActPerfReport::pc2);
    }
    if (counters > 3) {
        *reinterpret_cast<uint32_t *>(reinterpret_cast<uint32_t>(prPtr) + byteStep) = pr.pc3;
    }
}

#endif

} // namespace perf
} // namespace common_runtime
} // namespace nn

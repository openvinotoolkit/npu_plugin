//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/al/config/runtime.hpp"
#include "ze_api.h"
#include "ze_graph_ext.h"

#include <string>

#include <ie_common.h>

namespace vpux {
namespace zeroUtils {
std::string result_to_string(const ze_result_t result);

static inline void throwOnFail(const std::string& step, const ze_result_t result) {
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "throwOnFail: " << step << " result: " << result_to_string(result) << ", code 0x" << std::hex
                   << uint64_t(result);
    }
}

static inline ze_command_queue_priority_t toZeQueuePriority(const ov::hint::Priority& val) {
    switch (val) {
    case ov::hint::Priority::LOW:
        return ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
    case ov::hint::Priority::MEDIUM:
        return ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    case ov::hint::Priority::HIGH:
        return ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
    default:
        IE_THROW() << "Incorrect queue priority.";
    }
}

static inline std::size_t precisionToSize(const ze_graph_argument_precision_t val) {
    switch (val) {
    case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
        return 4;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
        return 2;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
        return 2;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
        return 1;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
        return 4;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
        return 2;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
        return 1;
    default:
        IE_THROW() << "precisionToSize switch->default reached";
    }
}

static inline ze_graph_argument_precision_t getZePrecision(const InferenceEngine::Precision precision) {
    switch (precision) {
    case InferenceEngine::Precision::I8:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT8;
    case InferenceEngine::Precision::U8:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT8;
    case InferenceEngine::Precision::I16:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT16;
    case InferenceEngine::Precision::U16:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT16;
    case InferenceEngine::Precision::I32:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT32;
    case InferenceEngine::Precision::FP16:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP16;
    case InferenceEngine::Precision::FP32:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP32;
    case InferenceEngine::Precision::BIN:
        return ZE_GRAPH_ARGUMENT_PRECISION_BIN;
    default:
        return ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN;
    }
}

static inline std::size_t layoutCount(const ze_graph_argument_layout_t val) {
    switch (val) {
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCHW:
        return 4;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NHWC:
        return 4;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW:
        return 5;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC:
        return 5;
    case ZE_GRAPH_ARGUMENT_LAYOUT_OIHW:
        return 4;
    case ZE_GRAPH_ARGUMENT_LAYOUT_C:
        return 1;
    case ZE_GRAPH_ARGUMENT_LAYOUT_CHW:
        return 3;
    case ZE_GRAPH_ARGUMENT_LAYOUT_HW:
        return 2;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NC:
        return 2;
    case ZE_GRAPH_ARGUMENT_LAYOUT_CN:
        return 2;
    default:
        IE_THROW() << "layoutCount switch->default reached";
    }
}

static inline std::size_t getSizeIOBytes(const ze_graph_argument_properties_t& argument) {
    std::size_t num_elements = 1;
    for (std::size_t i = 0; i < layoutCount(argument.deviceLayout); ++i) {
        num_elements *= argument.dims[i];
    }
    const std::size_t size_in_bytes = num_elements * precisionToSize(argument.devicePrecision);
    return size_in_bytes;
}

template <typename Map>
auto mapArguments(Map& zero, const std::string& key) -> typename Map::mapped_type& {
    for (auto& p : zero) {
        if (std::string::npos != p.first.find(key)) {
            return p.second;
        }
    }
    IE_THROW() << "mapArguments: fail to map";
}

template <typename Map>
auto mapArguments(const Map& zero, const std::string& key) -> const typename Map::mapped_type& {
    for (auto& p : zero) {
        if (std::string::npos != p.first.find(key)) {
            return p.second;
        }
    }
    IE_THROW() << "mapArguments: fail to map";
}

}  // namespace zeroUtils
}  // namespace vpux

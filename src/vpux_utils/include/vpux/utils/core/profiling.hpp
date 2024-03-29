//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <string>
#include <tuple>

namespace vpux {

// This char is a separator between original layer name provided in xml
// and metadata added by the compiler.
// It is crucial to provide layer names matching the original model in xml.
// This symbol must be unique in layer name.
constexpr char LOCATION_ORIGIN_SEPARATOR = '?';
constexpr char LOCATION_SEPARATOR = '/';

constexpr char PROFILING_CMX_2_DDR_OP_NAME[] = "ProfilingCMX2DDR";
constexpr char PROFILING_WORKPOINT_READ_ATTR[] = "PROFWORKPOINT_READ";
constexpr char PROFILING_OUTPUT_NAME[] = "profilingOutput";

// DMA HW profiling and workpoint capture require 64B section alignment
constexpr size_t PROFILING_SECTION_ALIGNMENT = 64;
constexpr size_t WORKPOINT_BUFFER_SIZE = 64;  // must be in a separate cache line

namespace profiling {

enum class ExecutorType { DPU = 1, UPA = 2, ACTSHAVE = 3, DMA_SW = 4, WORKPOINT = 5, DMA_HW = 6 };

std::string convertExecTypeToName(ExecutorType execType);
ExecutorType convertDataInfoNameToExecType(StringRef name);

template <typename T>
bool profilingTaskStartTimeComparator(const T& a, const T& b) {
    const auto namesCompareResult = std::strcmp(a.name, b.name);
    return std::forward_as_tuple(a.start_time_ns, namesCompareResult, b.duration_ns) <
           std::forward_as_tuple(b.start_time_ns, 0, a.duration_ns);
}
}  // namespace profiling
}  // namespace vpux

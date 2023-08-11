//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once
namespace vpux {

// This char is a separator between original layer name provided in xml
// and metadata added by the compiler.
// It is crucial to provide layer names matching the original model in xml.
// This symbol must be unique in layer name.
constexpr char LOCATION_ORIGIN_SEPARATOR = '?';
constexpr char LOCATION_SEPARATOR = '/';

constexpr char PROFILING_CMX_2_DDR_OP_NAME[] = "ProfilingCMX2DDR";
constexpr char PROFILING_DMA_TASK_BEGIN_PREFIX[] = "PROFTASKBEGIN";
constexpr char PROFILING_DMA_TASK_END_PREFIX[] = "PROFTASKEND";
constexpr char PROFILING_WORKPOINT_READ_ATTR[] = "PROFWORKPOINT_READ";
constexpr char PROFILING_PREFIX[] = "PROF";
}  // namespace vpux

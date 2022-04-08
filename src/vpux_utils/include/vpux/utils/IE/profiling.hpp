//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <ie_common.h>
#include "vpux/utils/plugin/profiling_parser.hpp"

namespace vpux {
namespace profiling {

enum class OutputType { NONE, TEXT, JSON, DEBUG };

// This function decodes profiling buffer into readable format.
// Format can be either regular text or TraceEvent json.
// outputFile is an optional argument - path to a file to store output. stdout if empty.
void outputWriter(const OutputType profilingType, const std::pair<const uint8_t*, size_t>& blob,
                  const std::pair<const uint8_t*, size_t>& profiling, const std::string& outputFile,
                  VerbosityLevel verbosity, bool fpga);

void printProfilingAsTraceEvent(const std::vector<TaskInfo>& taskProfiling,
                                const std::vector<LayerInfo>& layerProfiling, std::ostream& out_stream);

void printProfilingAsText(const std::vector<TaskInfo>& taskProfiling, const std::vector<LayerInfo>& layerProfiling,
                          std::ostream& out_stream);

void printDebugProfilingInfo(const std::vector<DebugInfo>& debugProfiling, std::ostream& out_stream);

void printSummary(const SummaryInfo& summary, std::ostream& out_stream);

bool isClusterLevelProfilingTask(const TaskInfo& task);

bool isLowLevelProfilingTask(const TaskInfo& task);

}  // namespace profiling
}  // namespace vpux

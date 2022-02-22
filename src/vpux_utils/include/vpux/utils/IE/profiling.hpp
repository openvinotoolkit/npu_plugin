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

#include <ie_common.h>
#include "vpux/utils/plugin/profiling_parser.hpp"

namespace vpux {
namespace profiling {

enum class OutputType { NONE, TEXT, JSON };

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> convertProfilingLayersToIEInfo(
        std::vector<LayerInfo>& layerInfo);

// This function decodes profiling buffer into readable format.
// Format can be either regular text or TraceEvent json.
// outputFile is an optional argument - path to a file to store output. stdout if empty.
void outputWriter(const OutputType profilingType, const std::pair<const uint8_t*, uint64_t>& blob,
                  const std::pair<const uint8_t*, uint64_t>& profiling, const std::string& outputFile);

}  // namespace profiling
}  // namespace vpux

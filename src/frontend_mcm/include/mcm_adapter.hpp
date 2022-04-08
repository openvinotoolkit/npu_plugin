//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux_compiler.hpp"

#include <schema/graphfile/graphfile_generated.h>

#include <ie_core.hpp>

namespace vpu {
namespace MCMAdapter {

struct MetaInfo {
    std::string _networkName;
    InferenceEngine::InputsDataMap _inputs;
    InferenceEngine::OutputsDataMap _outputs;
    vpux::QuantizationParamMap _quantParams;
};

bool isMCMCompilerAvailable();

/**
 * @brief Deserialization meta data from graph blob
 * @param header The header of the graph blob struct
 * @param config Compiler config
 * @return Meta data from graph blob (network name, IE inputs/outputs)
 */
MetaInfo deserializeMetaData(const MVCNN::SummaryHeader& header, const vpux::Config& config);
}  // namespace MCMAdapter
}  // namespace vpu

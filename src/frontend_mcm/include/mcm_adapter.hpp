//
// Copyright 2020 Intel Corporation.
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

#include <ie_core.hpp>
#include <mcm_config.hpp>

#include <schema/graphfile/graphfile_generated.h>

namespace vpu {
namespace MCMAdapter {

struct MetaInfo {
    std::string _networkName;
    InferenceEngine::InputsDataMap _inputs;
    InferenceEngine::OutputsDataMap _outputs;
};

bool isMCMCompilerAvailable();

/**
 * @brief Deserialization meta data from graph blob
 * @param graphFileInstance The struct with graph blob data
 * @param config Compiler config
 * @return Meta data from graph blob (network name, IE inputs/outputs)
 */
MetaInfo deserializeMetaData(const MVCNN::GraphFileT& graphFileInstance, const MCMConfig& config);
}  // namespace MCMAdapter
}  // namespace vpu

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

namespace vpu {
namespace MCMAdapter {

struct MetaInfo {
    std::string _networkName;
    InferenceEngine::InputsDataMap _inputs;
    InferenceEngine::OutputsDataMap _outputs;
};

bool isMCMCompilerAvailable();

MetaInfo deserializeMetaData(const std::vector<char>& outBlob, const MCMConfig& config);
}  // namespace MCMAdapter
}  // namespace vpu

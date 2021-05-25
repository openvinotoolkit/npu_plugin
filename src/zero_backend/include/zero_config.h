//
// Copyright 2021 Intel Corporation.
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

#include <map>
#include <string>
#include <unordered_set>
#include <vpux_config.hpp>
#include "zero_private_config.h"

namespace vpux {

class ZeroConfig final : public vpux::VPUXConfig {
public:
    ZeroConfig();

    InferenceEngine::VPUXConfigParams::ze_syncType ze_syncType() const {
        return _ze_syncType;
    }

protected:
    void parse(const std::map<std::string, std::string>& config) override;

private:
    InferenceEngine::VPUXConfigParams::ze_syncType _ze_syncType =
            InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT;
};
}  // namespace vpux

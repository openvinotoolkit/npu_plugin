//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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

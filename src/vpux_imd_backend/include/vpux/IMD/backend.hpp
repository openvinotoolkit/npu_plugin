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

#include "vpux.hpp"

namespace vpux {
namespace IMD {

class BackendImpl final : public IEngineBackend {
public:
    const std::shared_ptr<IDevice> getDevice() const override;
    const std::shared_ptr<IDevice> getDevice(const std::string& name) const override;
    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& params) const override;

    const std::vector<std::string> getDeviceNames() const override;

    const std::string getName() const override;

    void registerOptions(OptionsDesc& options) const override;
};

}  // namespace IMD
}  // namespace vpux

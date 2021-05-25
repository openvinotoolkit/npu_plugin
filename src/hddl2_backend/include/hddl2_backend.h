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
// Plugin
#include <vpux.hpp>
#include <vpux_config.hpp>

namespace vpux {
namespace hddl2 {
class HDDL2Backend final : public vpux::IEngineBackend {
public:
    using Ptr = std::shared_ptr<HDDL2Backend>;
    using CPtr = std::shared_ptr<const HDDL2Backend>;

    explicit HDDL2Backend();

    const std::shared_ptr<IDevice> getDevice() const override;
    const std::shared_ptr<IDevice> getDevice(const std::string& specificDeviceName) const override;
    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& map) const override;
    const std::vector<std::string> getDeviceNames() const override;
    const std::string getName() const override {
        return "HDDL2";
    }
    std::unordered_set<std::string> getSupportedOptions() const override {
        return _config.getRunTimeOptions();
    }

    // TODO remove static and make them private
    static bool isServiceAvailable(const vpu::Logger::Ptr& logger = nullptr);
    static bool isServiceRunning();

private:
    VPUXConfig _config;
    vpu::Logger::Ptr _logger = nullptr;
    std::map<std::string, std::shared_ptr<IDevice>> _devices;
    std::map<std::string, std::shared_ptr<IDevice>> createDeviceMap();
};
}  // namespace hddl2
}  // namespace vpux

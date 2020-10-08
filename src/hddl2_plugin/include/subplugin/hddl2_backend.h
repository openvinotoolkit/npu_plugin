//
// Copyright 2020 Intel Corporation.
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
#include <vpux.hpp>

namespace vpux {
namespace HDDL2 {
class HDDL2Backend final : public vpux::IEngineBackend {
public:
    using Ptr = std::shared_ptr<HDDL2Backend>;
    using CPtr = std::shared_ptr<const HDDL2Backend>;

    explicit HDDL2Backend(const VPUXConfig& config = {});

    const std::shared_ptr<IDevice> getDevice() const override;
    const std::shared_ptr<IDevice> getDevice(const std::string& specificDeviceName) const override;
    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& map) const override;
    const std::vector<std::string> getDeviceNames() const override;
    const std::string getName() const override { return "HDDL2"; }

    // TODO remove static and make them private
    static bool isServiceAvailable(const vpu::Logger::Ptr& logger = nullptr);
    static bool isServiceRunning();

private:
    vpu::Logger::Ptr _logger = nullptr;
    const std::map<std::string, std::shared_ptr<IDevice>> _devices;
    std::map<std::string, std::shared_ptr<IDevice>> createDeviceMap();
};
}  // namespace HDDL2
}  // namespace vpux

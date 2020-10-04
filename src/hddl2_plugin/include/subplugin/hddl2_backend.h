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

    HDDL2Backend(const VPUXConfig& config);
    /** @brief HDDL2 is considered as one device, so no separate devices will be returned */
    const std::map<std::string, std::shared_ptr<IDevice>>& getDevices() const override { return _devices; }
    /** @brief Search for specific device, if required*/
    const std::shared_ptr<IDevice> getDevice(const std::string& deviceName) const;

private:
    vpu::Logger::Ptr _logger;
    const std::map<std::string, std::shared_ptr<IDevice>> _devices;
    std::map<std::string, std::shared_ptr<IDevice>> createDeviceMap();
};
}  // namespace HDDL2
}  // namespace vpux

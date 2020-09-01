#pragma once

#include <ie_common.h>

#include <map>
#include <memory>
#include <vpux.hpp>

namespace vpux {

class VpualEngineBackend final : public vpux::IEngineBackend {
    std::unique_ptr<vpu::Logger> _logger;
    std::map<std::string, std::shared_ptr<IDevice>> _devices;

public:
    VpualEngineBackend();
    const std::map<std::string, std::shared_ptr<IDevice>>& getDevices() const override;

private:
    const std::map<std::string, std::shared_ptr<IDevice>> createDeviceMap();
};

}  // namespace vpux

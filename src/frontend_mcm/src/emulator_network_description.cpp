#include "emulator_network_description.hpp"

#include <utility>

namespace vpu {
namespace MCMAdapter {

EmulatorNetworkDescription::EmulatorNetworkDescription(const std::vector<char>& compiledNetwork,
                                                       const vpu::MCMConfig& config, const std::string& name)
        : _name{name},
          _logger{std::unique_ptr<vpu::Logger>(
                  new vpu::Logger("EmulatorNetworkDescription", config.logLevel(), consoleOutput()))},
          _dataMapPlaceholder{},
          _compiledNetwork{compiledNetwork},
          _quantParams{} {
    IE_ASSERT(!_compiledNetwork.empty());
}

const vpux::DataMap& EmulatorNetworkDescription::getInputsInfo() const {
    _logger->info("EmulatorNetworkDescription::getInputsInfo()\n");
    return _dataMapPlaceholder;
}

const vpux::DataMap& EmulatorNetworkDescription::getOutputsInfo() const {
    _logger->info("EmulatorNetworkDescription::getOutputsInfo()\n");
    return _dataMapPlaceholder;
}

const vpux::DataMap& EmulatorNetworkDescription::getDeviceInputsInfo() const {
    _logger->info("EmulatorNetworkDescription::getDeviceInputsInfo()\n");
    return _dataMapPlaceholder;
}

const vpux::DataMap& EmulatorNetworkDescription::getDeviceOutputsInfo() const {
    _logger->info("EmulatorNetworkDescription::getDeviceOutputsInfo()\n");
    return _dataMapPlaceholder;
}

const vpux::DataMap& EmulatorNetworkDescription::getDeviceProfilingOutputsInfo() const {
    _logger->info("EmulatorNetworkDescription::getDeviceProfilingsInfo()\n");
    return _dataMapPlaceholder;
}

const vpux::QuantizationParamMap& EmulatorNetworkDescription::getQuantParamsInfo() const {
    _logger->info("EmulatorNetworkDescription::getQuantParamsInfo()\n");
    return _quantParams;
}

const std::vector<char>& EmulatorNetworkDescription::getCompiledNetwork() const {
    return _compiledNetwork;
}

const void* EmulatorNetworkDescription::getNetworkModel() const {
    return _compiledNetwork.data();
}

std::size_t EmulatorNetworkDescription::getNetworkModelSize() const {
    return _compiledNetwork.size();
}

const std::string& EmulatorNetworkDescription::getName() const {
    return _name;
}

}  // namespace MCMAdapter
}  // namespace vpu

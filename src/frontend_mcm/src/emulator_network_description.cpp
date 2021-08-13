#include "emulator_network_description.hpp"

#include <utility>

namespace vpu {
namespace MCMAdapter {

EmulatorNetworkDescription::EmulatorNetworkDescription(const std::vector<char>& compiledNetwork,
                                                       const vpu::MCMConfig& config, const std::string& name)
        : _name{name},
          _compiler{nullptr},
          _logger{std::unique_ptr<vpu::Logger>(
                  new vpu::Logger("EmulatorNetworkDescription", config.logLevel(), consoleOutput()))},
          _dataMapPlaceholder{},
          _compiledNetwork{compiledNetwork} {
}

const vpux::DataMap& EmulatorNetworkDescription::getInputsInfo() const {
    _logger->warning("EmulatorNetworkDescription::getInputsInfo()\n");
    return _dataMapPlaceholder;
}

const vpux::DataMap& EmulatorNetworkDescription::getOutputsInfo() const {
    _logger->warning("EmulatorNetworkDescription::getOutputsInfo()\n");
    return _dataMapPlaceholder;
}

const vpux::DataMap& EmulatorNetworkDescription::getDeviceInputsInfo() const {
    _logger->warning("EmulatorNetworkDescription::getDeviceInputsInfo()\n");
    return _dataMapPlaceholder;
}

const vpux::DataMap& EmulatorNetworkDescription::getDeviceOutputsInfo() const {
    _logger->warning("EmulatorNetworkDescription::getDeviceOutputsInfo()\n");
    return _dataMapPlaceholder;
}

const std::vector<char>& EmulatorNetworkDescription::getCompiledNetwork() const {
    if (_compiledNetwork.empty())
        _logger->warning("EmulatorNetworkDescription::getCompiledNetwork() - _compiledNetwork is empty\n");
    return _compiledNetwork;
}

const void* EmulatorNetworkDescription::getNetworkModel() const {
    if (_compiledNetwork.empty())
        _logger->warning("EmulatorNetworkDescription::getCompiledNetwork() - _compiledNetwork is empty\n");
    return _compiledNetwork.data();
}

std::size_t EmulatorNetworkDescription::getNetworkModelSize() const {
    if (_compiledNetwork.empty())
        return sizeof _compiler->model();

    return _compiledNetwork.size();
}

const std::string& EmulatorNetworkDescription::getName() const {
    return _name;
}

}  // namespace MCMAdapter
}  // namespace vpu

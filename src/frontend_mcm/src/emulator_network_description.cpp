#include "emulator_network_description.hpp"

#include <utility>

namespace vpu {
namespace MCMAdapter {

EmulatorNetworkDescription::EmulatorNetworkDescription(std::unique_ptr<mv::CompilationUnit>&& compiler,
                                                       const vpu::MCMConfig& config, const std::string& name)
        : _name{name},
          _compiler{std::move(compiler)},
          _logger{std::unique_ptr<vpu::Logger>(
                  new vpu::Logger("EmulatorNetworkDescription", config.logLevel(), consoleOutput()))} {
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
    _logger->warning("EmulatorNetworkDescription::getCompiledNetwork()\n");
    return _compiledNetworkPlaceholder;
}

const void* EmulatorNetworkDescription::getNetworkModel() const {
    return &_compiler->model();
}

std::size_t EmulatorNetworkDescription::getNetworkModelSize() const {
    return sizeof _compiler->model();
}

const std::string& EmulatorNetworkDescription::getName() const {
    return _name;
}

}  // namespace MCMAdapter
}  // namespace vpu

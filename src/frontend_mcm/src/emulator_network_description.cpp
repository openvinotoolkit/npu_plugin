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

#include "emulator_network_description.hpp"

#include "vpux/al/config/common.hpp"

#include <utility>

using namespace vpux;

namespace vpu {
namespace MCMAdapter {

EmulatorNetworkDescription::EmulatorNetworkDescription(const std::vector<char>& compiledNetwork, const Config& config,
                                                       const std::string& name)
        : _name{name},
          _logger("EmulatorNetworkDescription", config.get<LOG_LEVEL>()),
          _dataMapPlaceholder{},
          _compiledNetwork{compiledNetwork},
          _quantParams{},
          _ovParameters{},
          _ovResults{} {
    IE_ASSERT(!_compiledNetwork.empty());
}

const DataMap& EmulatorNetworkDescription::getInputsInfo() const {
    _logger.info("EmulatorNetworkDescription::getInputsInfo()");
    return _dataMapPlaceholder;
}

const DataMap& EmulatorNetworkDescription::getOutputsInfo() const {
    _logger.info("EmulatorNetworkDescription::getOutputsInfo()");
    return _dataMapPlaceholder;
}

const DataMap& EmulatorNetworkDescription::getDeviceInputsInfo() const {
    _logger.info("EmulatorNetworkDescription::getDeviceInputsInfo()");
    return _dataMapPlaceholder;
}

const DataMap& EmulatorNetworkDescription::getDeviceOutputsInfo() const {
    _logger.info("EmulatorNetworkDescription::getDeviceOutputsInfo()");
    return _dataMapPlaceholder;
}

const DataMap& EmulatorNetworkDescription::getDeviceProfilingOutputsInfo() const {
    _logger.info("EmulatorNetworkDescription::getDeviceProfilingsInfo()");
    return _dataMapPlaceholder;
}

const std::vector<OVRawNode>& EmulatorNetworkDescription::getOVParameters() const {
    _logger.info("EmulatorNetworkDescription::getOVParameters()");
    return _ovParameters;
}

const std::vector<OVRawNode>& EmulatorNetworkDescription::getOVResults() const {
    _logger.info("EmulatorNetworkDescription::getOVResults()");
    return _ovResults;
}

const QuantizationParamMap& EmulatorNetworkDescription::getQuantParamsInfo() const {
    _logger.info("EmulatorNetworkDescription::getQuantParamsInfo()");
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

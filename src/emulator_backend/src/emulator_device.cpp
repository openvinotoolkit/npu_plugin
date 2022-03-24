//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "emulator_device.hpp"

#include "emulator_executor.hpp"

#include <memory>

namespace ie = InferenceEngine;

namespace vpux {

EmulatorDevice::EmulatorDevice(): _logger("EmulatorBackend", LogLevel::Debug /*_config.logLevel()*/) {
}

std::shared_ptr<Executor> EmulatorDevice::createExecutor(const NetworkDescription::Ptr& network, const Config& config) {
    _logger.debug("::createExecutor() started");
    if (network->getNetworkModel() == nullptr)
        IE_THROW() << "Network passed to emulator is incorrect";
    _logger.debug("::createExecutor() finished");
    return std::make_shared<EmulatorExecutor>(network, config);
}

std::string EmulatorDevice::getName() const {
    return "EMULATOR";
}

}  // namespace vpux

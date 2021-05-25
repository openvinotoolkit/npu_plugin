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

#include <memory>

#include <include/mcm/compiler/compilation_unit.hpp>
#include <vpux.hpp>

#include "emulator_device.hpp"
#include "emulator_executor.hpp"

namespace ie = InferenceEngine;

namespace vpux {

EmulatorDevice::EmulatorDevice()
        : _logger(std::unique_ptr<vpu::Logger>(new vpu::Logger(
                  "EmulatorBackend", vpu::LogLevel::Debug /*_config.logLevel()*/, vpu::consoleOutput()))) {
}

std::shared_ptr<Executor> EmulatorDevice::createExecutor(const NetworkDescription::Ptr& network,
                                                         const VPUXConfig& /*config*/) {
    _logger->debug("::createExecutor() started");
    if (network->getNetworkModel() == nullptr)
        IE_THROW() << "Network passed to emulator is incorrect";
    _logger->debug("::createExecutor() finished");
    return std::make_shared<EmulatorExecutor>(network);
}  // namespace vpux

std::string EmulatorDevice::getName() const {
    return "EMULATOR";
}

}  // namespace vpux

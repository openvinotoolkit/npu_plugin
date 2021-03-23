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

//
// Copyright 2019-2020 Intel Corporation.
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

#include "emulator_executor.hpp"

#include "vpux/al/config/common.hpp"

#include <emu/arithmetic_types.hpp>

#include <file_utils.h>

namespace ie = InferenceEngine;

namespace vpux {

EmulatorExecutor::EmulatorExecutor(const vpux::NetworkDescription::Ptr& network, const Config& config)
        : _logger("EmulatorBackend", LogLevel::Debug /*_config.logLevel()*/),
          _network(network),
          _manager(ie::getIELibraryPath() + "/vpux_emulator", config.get<LOG_LEVEL>()) {
}

void EmulatorExecutor::push(const ie::BlobMap& inputs, const PreprocMap&) {
    push(inputs);
}

void EmulatorExecutor::push(const ie::BlobMap& inputs) {
    _logger.debug("EmulatorExecutor::push() started");
    _manager.reset(_network->getNetworkModel());
    auto inputIt = inputs.cbegin();
    for (const auto inputName : _manager.getNetworkInputs()) {
        const ie::Blob& blob = *inputIt->second;
        _manager.populate(inputName, blob.cbuffer().as<const void*>());
        ++inputIt;
    }
    _manager.run();
    _logger.debug("EmulatorExecutor::push() finished");
}

void EmulatorExecutor::pull(ie::BlobMap& outputs) {
    _logger.debug("EmulatorExecutor::pull() started");
    auto outputIt = outputs.begin();
    for (const auto outputName : _manager.getNetworkOutputs()) {
        ie::Blob& blob = *outputIt->second;
        std::copy_n(_manager.data(outputName).cbegin(), blob.byteSize(), blob.buffer().as<char*>());
        ++outputIt;
    }
    _logger.debug("EmulatorExecutor::pull() finished");
}

}  // namespace vpux

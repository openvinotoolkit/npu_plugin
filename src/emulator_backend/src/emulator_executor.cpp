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

#include <file_utils.h>
#include <emu/arithmetic_types.hpp>

namespace ie = InferenceEngine;

namespace vpux {

EmulatorExecutor::EmulatorExecutor(const vpux::NetworkDescription::Ptr& network)
        : _logger("EmulatorBackend", vpu::LogLevel::Debug /*_config.logLevel()*/, vpu::consoleOutput()),
          _network(network),
          _manager(ie::getIELibraryPath() + "/mcm_emulator") {
}

void EmulatorExecutor::push(const ie::BlobMap& inputs, const PreprocMap&) {
    push(inputs);
}

void EmulatorExecutor::push(const ie::BlobMap& inputs) {
    _logger.debug("EmulatorExecutor::push() started");
    _manager.reset(*reinterpret_cast<mv::OpModel*>(const_cast<void*>(_network->getNetworkModel())));
    auto inputIt = inputs.cbegin();
    for (const auto inputOp : _manager.opModel().getNetworkInputs()) {
        const ie::Blob& blob = *inputIt->second;
        const mv::Tensor& tensor = *inputOp->getOutputTensor(0);
        _manager.populate(tensor, tensor.getOrder(), blob.cbuffer().as<const void*>());
        ++inputIt;
    }
    _manager.run();
    _logger.debug("EmulatorExecutor::push() finished");
}

void EmulatorExecutor::pull(ie::BlobMap& outputs) {
    _logger.debug("EmulatorExecutor::pull() started");
    auto outputIt = outputs.begin();
    for (const auto outputOp : _manager.opModel().getNetworkOutputs()) {
        ie::Blob& blob = *outputIt->second;
        std::copy_n(_manager.data(*outputOp->getInputTensor(0)).cbegin(), blob.byteSize(), blob.buffer().as<char*>());
        ++outputIt;
    }
    _logger.debug("EmulatorExecutor::pull() finished");
}

}  // namespace vpux

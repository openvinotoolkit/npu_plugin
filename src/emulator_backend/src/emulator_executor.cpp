//
// Copyright 2019-2020 Intel Corporation.
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

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

#include "vpux/compiler/compiler.hpp"

#include "vpux/compiler/backend/VPUIP.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/network_description.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/frontend/IE.hpp"
#include "vpux/compiler/pipelines.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/helper_macros.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

#include <cpp/ie_cnn_network.h>
#include <description_buffer.hpp>

#include <algorithm>

using namespace vpux;
using namespace InferenceEngine;

std::shared_ptr<INetworkDescription> vpux::CompilerImpl::compile(ICNNNetwork&, const VPUXConfig&) {
    VPUX_THROW("VPUX Compiler doesn't support InferenceEngine IR prior to v10 "
               "version");
}

namespace {

LogLevel getLogLevel(const VPUXConfig& config) {
    switch (config.logLevel()) {
    case vpu::LogLevel::Fatal:
        return LogLevel::Fatal;
    case vpu::LogLevel::Error:
        return LogLevel::Error;
    case vpu::LogLevel::Warning:
        return LogLevel::Warning;
    case vpu::LogLevel::Info:
        return LogLevel::Info;
    case vpu::LogLevel::Debug:
        return LogLevel::Debug;
    case vpu::LogLevel::Trace:
        return LogLevel::Trace;
    default:
        return LogLevel::None;
    }
}

}  // namespace

std::shared_ptr<INetworkDescription> vpux::CompilerImpl::compile(const std::shared_ptr<ngraph::Function>& func,
                                                                 const std::string&, const InputsDataMap& inputsInfo,
                                                                 const OutputsDataMap& outputsInfo,
                                                                 const VPUXConfig& config) {
    Logger log("vpux-compiler", getLogLevel(config));

    CNNNetwork cnnNet(func);

    for (const auto& p : inputsInfo) {
        cnnNet.getInputsInfo().at(p.first)->setPrecision(p.second->getPrecision());
        cnnNet.getInputsInfo().at(p.first)->setLayout(p.second->getLayout());
    }
    for (const auto& p : outputsInfo) {
        cnnNet.getOutputsInfo().at(p.first)->setPrecision(p.second->getPrecision());
        cnnNet.getOutputsInfo().at(p.first)->setLayout(p.second->getLayout());
    }

    mlir::MLIRContext ctx;
    addLogging(ctx, log);

    ctx.loadDialect<IE::IEDialect>();
    ctx.loadDialect<IERT::IERTDialect>();
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    IE::FrontEnd frontEnd(&ctx, log);
    auto module = frontEnd.importNetwork(cnnNet);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    addLogging(pm, log);

    pm.addPass(createReferenceModePass(log.nest()));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module.get())), "Compilation failed");

    const auto blob = VPUIP::exportToBlob(module.get(), log);

    std::vector<char> compiledNetwork(blob.size());
    std::copy_n(reinterpret_cast<const char*>(blob.data()), blob.size(), compiledNetwork.data());

    return std::make_shared<VPUIP::NetworkDescription>(std::move(compiledNetwork));
}

std::shared_ptr<vpux::INetworkDescription> vpux::CompilerImpl::parse(const std::vector<char>& compiledNetwork,
                                                                     const vpux::VPUXConfig&, const std::string&) {
    return std::make_shared<VPUIP::NetworkDescription>(compiledNetwork);
}

std::set<std::string> vpux::CompilerImpl::getSupportedLayers(InferenceEngine::ICNNNetwork&) {
    VPUX_THROW("VPUX Compiler doesn't support HETERO mode");
}

std::unordered_set<std::string> vpux::CompilerImpl::getSupportedOptions() {
    return {};
}

INFERENCE_PLUGIN_API(StatusCode)
CreateVPUXCompiler(ICompiler*& compiler, ResponseDesc* resp) noexcept {
    try {
        compiler = new CompilerImpl();
        return StatusCode::OK;
    } catch (const std::exception& ex) {
        return DescriptionBuffer(StatusCode::GENERAL_ERROR, resp) << ex.what();
    }
}

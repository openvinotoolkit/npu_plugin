//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/Timing.h>

#include <cpp/ie_cnn_network.h>
#include <vpux_compiler.hpp>

namespace vpux {
namespace IE {

mlir::OwningModuleRef importNetwork(mlir::MLIRContext* ctx, InferenceEngine::CNNNetwork cnnNet,
                                    std::vector<PreProcessInfo>& preProcInfo, bool sharedConstants,
                                    mlir::TimingScope& rootTiming, bool enableProfiling, vpux::VPU::ArchKind arch,
                                    Logger log = Logger::global());

std::unordered_set<std::string> queryNetwork(const InferenceEngine::CNNNetwork& cnnNet,
                                             std::vector<PreProcessInfo>& preProcInfo, mlir::TimingScope& rootTiming,
                                             const vpux::VPU::ArchKind arch, Logger log = Logger::global());

}  // namespace IE
}  // namespace vpux

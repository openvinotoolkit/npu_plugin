//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/dialect.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <memory>

namespace vpux {
namespace VPU {

//
// Passes
//

std::unique_ptr<mlir::Pass> createInitCompilerPass();
std::unique_ptr<mlir::Pass> createInitCompilerPass(ArchKind arch, CompilationMode compilationMode,
                                                   Optional<int> numOfDPUGroups = None, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createCMXConcatPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitNCEOpsOntoWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapVPUOpsInNCEClusterTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustMemorySpacePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMultiClusterStrategyAssignmentPass(Logger log = Logger::global());

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/VPU/generated/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/VPU/generated/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace VPU
}  // namespace vpux

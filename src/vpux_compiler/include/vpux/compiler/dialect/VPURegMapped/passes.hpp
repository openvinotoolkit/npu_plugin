//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace vpux {
namespace VPURegMapped {

//
// Passes
//

std::unique_ptr<mlir::Pass> resolveMappedInferenceTaskLocationsPass(Logger log = Logger::global());

// pass object stores callable, so it cannot llvm::function_ref
using UpperBoundsCallable = std::function<size_t(VPURegMapped::TaskType, VPURegMapped::IndexType)>;
std::unique_ptr<mlir::Pass> resolveMappedInferenceTaskLocationsPass(UpperBoundsCallable upperBounds,
                                                                    Logger log = Logger::global());

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/VPURegMapped/generated/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/VPURegMapped/generated/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace VPURegMapped
}  // namespace vpux

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/EMU/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>
#include <type_traits>

namespace vpux {
namespace EMU {

//
// LowPrecision
//

void buildAdjustForEMU(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createSqueezeBiasShapePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustFQPrecisionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddWeightsTableToEmuPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createRemoveWeightsAlignmentPass(Logger log = Logger::global());

//
// Registration
//

void registerEMUPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/EMU/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/EMU/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace EMU
}  // namespace vpux

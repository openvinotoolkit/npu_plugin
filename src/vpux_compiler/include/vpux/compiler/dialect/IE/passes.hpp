//
// Copyright Intel Corporation.
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

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <memory>

namespace vpux {
namespace IE {

//
// Passes
//

std::unique_ptr<mlir::Pass> createUseUserPrecisionPass(Logger log = Logger::global());

//
// Adjust IE Dialect IR for VPU target.
//
// This pipeline includes various adaptation passes to adjust the IR for VPU target.
//

void buildAdjustForVPUPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertFCToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertTile2PerAxisTilePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPrecisionToFP16Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertShapeTo4DPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertPaddingsToFloorModePass(Logger log = Logger::global());

//
// Low precision transformations.
//
// This pipeline includes all transformations to support low precisions.
//

void buildLowPrecisionPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createSplitFakeQuantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createQuantizeConstPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDequantizeConstPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMergeFakeQuantPass(Logger log = Logger::global());

//
// Registration
//

void registerPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/IE/generated/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/IE/generated/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace IE
}  // namespace vpux

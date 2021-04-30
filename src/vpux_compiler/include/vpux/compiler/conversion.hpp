//
// Copyright Intel Corporation.
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

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <memory>

namespace vpux {

//
// Performs full lowering from the IE Dialect to IERT Dialect.
//
// This pipeline performs full IR lowering from IE Dialect to IERT Dialect,
// including Function types, call graph and return operations.
//

void buildLowerIE2IERTPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createBufferizeIEPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddBuffersForNetResults(Logger log = Logger::global());

//
// Performs partial lowering from the IERT Dialect to VPUIP Dialect.
//
// Converts IERT Operations to VPUIP NCE Operations, where possible.
//

std::unique_ptr<mlir::Pass> createConvertToNCEOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseActivationsPass(Logger log = Logger::global());

//
// Performs full lowering from the IERT Dialect to VPUIP Dialect.
//
// Replaces Layers with VPUIP UPA/DMA tasks counterparts and declarations with VPUIP analogues.
//

void buildLowerIERT2VPUIPPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertLayers2VPUIPPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDeclarations2VPUIPPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertViewOps2VPUIPPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAsyncOps2VPUIPPass(Logger log = Logger::global());

//
// Registration
//

void registerConversionPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/conversion/generated/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/conversion/generated/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace vpux

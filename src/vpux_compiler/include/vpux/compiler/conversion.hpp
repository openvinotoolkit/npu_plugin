//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/EMU/ops.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/pipelines.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <memory>

namespace vpux {

//
// LowerIE2IERT
//

//
// Performs full lowering from the IE Dialect to IERT Dialect.
//
// This pipeline performs full IR lowering from IE Dialect to IERT Dialect,
// including Function types, call graph and return operations.
//

void buildLowerIE2IERTPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createBufferizeIEPass(Logger log = Logger::global());

//
// LowerIE2VPU
//

std::unique_ptr<mlir::Pass> createConvertLayers2VPUPass(Logger log = Logger::global());

//
// LowerVPU2VPUIP
//

//
// Performs full lowering from the VPU Dialect to VPUIP Dialect.
//
// This pipeline performs full IR lowering from VPU Dialect to VPUIP Dialect,
// including Function types, call graph and return operations.
//

std::unique_ptr<mlir::Pass> createBufferizeFuncAndReturnPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddBuffersForNetResults(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertVPUNCEToVPUIPPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertNCEClusterTilingToVPUIPPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertSWLayers2VPUIPSWKernelPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSWLayers2VPUIPUPAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSWLayers2AffinePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAffine2LLVMPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertLayers2VPUIPPass(Logger log = Logger::global());

// ELF back-end lowerings
std::unique_ptr<mlir::Pass> createConvertVPUIP2VPUMI37XXPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertVPUMI37XX2ELFPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createMoveIOBuffersToSectionsPass(Logger log = Logger::global());

//
// LowerVPUNCE2EMU
//

//
// Performs full lowering from the VPU Dialect to EMU Dialect.
//
std::unique_ptr<mlir::Pass> createConvertVPUNCEToEMUPass(Logger log = Logger::global());

//
// registerConversionPipelines
//

void registerConversionPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/conversion/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/conversion/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace vpux

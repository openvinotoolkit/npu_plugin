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

#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/PassManager.h>

namespace vpux {

void registerPipelines();

//
// Compiles IE Network in Reference mode (SW only execution).
//
// This pipeline performs full IR lowering from IE Dialect to VPUIP Dialect (using intermediate lowerings).
// It uses simple SW implementation for layer operations without any optimizations.
//

void buildReferenceModePipeline(mlir::OpPassManager& pm, bool enableProfiling = false, Logger log = Logger::global());
void buildHardwareModePipeline(mlir::OpPassManager& pm, bool enableProfiling = false, Logger log = Logger::global());

}  // namespace vpux

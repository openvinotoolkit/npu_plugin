//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion.hpp"

namespace vpux {
namespace arch30xx {

//
// LowerIE2VPU
//

std::unique_ptr<mlir::Pass> createConvertIEToVPUNCEPass(Logger log = Logger::global());

//
// pipelines
//

void buildLowerIE2VPUPipeline30XX(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildLowerVPU2VPUIP30XXPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

//
// registerConversionPipeline30XX
//

void registerConversionPipeline30XX();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/VPU30XX/conversion/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/VPU30XX/conversion/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace arch30xx
}  // namespace vpux

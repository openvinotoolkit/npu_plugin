//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/passes.hpp"

namespace vpux {
namespace VPU {
namespace arch37xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createAdjustForOptimizedSwKernelPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitRealDFTOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDecomposeGatherPass(Logger log = Logger::global());

void buildIncrementalPipeline(mlir::OpPassManager& pm, const VPU::TilingOptions& options,
                              Logger log = Logger::global());
void registerVPUPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/VPU37XX/dialect/VPU/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/VPU37XX/dialect/VPU/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace arch37xx
}  // namespace VPU
}  // namespace vpux

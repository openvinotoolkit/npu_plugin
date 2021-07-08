//
// Copyright 2021 Intel Corporation.
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

#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/Pass/Pass.h>
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/edsl/autotile.hpp"

namespace vpux {
namespace edsl {

//
// Passes
//

std::unique_ptr<mlir::Pass> createAutoTileVPUXPass(const AutoTileParams& params);
std::unique_ptr<mlir::Pass> createAutoTileVPUXPass();
std::unique_ptr<mlir::Pass> createShavePatternsPass();
std::unique_ptr<mlir::Pass> createShavePipelinePass();
std::unique_ptr<mlir::Pass> createSinkScalarPass();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/edsl/generated/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/edsl/generated/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace edsl
}  // namespace vpux

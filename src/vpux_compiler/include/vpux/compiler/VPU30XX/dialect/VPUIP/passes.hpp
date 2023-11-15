//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

namespace vpux {
namespace VPUIP {
namespace arch30xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createUnrollClusterTilingPass(Logger log = Logger::global());

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/VPU30XX/dialect/VPUIP/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/VPU30XX/dialect/VPUIP/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace arch30xx
}  // namespace VPUIP
}  // namespace vpux

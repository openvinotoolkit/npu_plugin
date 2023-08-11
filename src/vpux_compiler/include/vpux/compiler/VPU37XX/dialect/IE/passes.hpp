//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/passes.hpp"

namespace vpux {
namespace IE {

//
// Passses
//

std::unique_ptr<mlir::Pass> createDeletePerAxisQuantizationPass(Logger log = Logger::global());

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/VPU37XX/dialect/IE/generated/passes.hpp.inc>
#undef GEN_PASS_CLASSES

// An additional namespace to avoid redefinition of registerIEPasses method
namespace Arch37XX {

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/VPU37XX/dialect/IE/generated/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace Arch37XX
}  // namespace IE
}  // namespace vpux

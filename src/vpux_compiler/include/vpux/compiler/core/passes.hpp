//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>

namespace vpux {

//
// Passes
//

std::unique_ptr<mlir::Pass> createMoveDeclarationsToTopPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPrintDotPass(StringRef fileName = {}, StringRef startAfter = {},
                                               StringRef stopBefore = {}, bool printConst = false,
                                               bool printDeclarations = false);

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/core/generated/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/core/generated/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace vpux

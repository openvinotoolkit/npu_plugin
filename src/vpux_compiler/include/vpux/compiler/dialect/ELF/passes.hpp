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

//#include "vpux/compiler/dialect/ELF/attributes/enums.hpp"
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace ELF {

//
// Passes
//
std::unique_ptr<mlir::Pass> createSetCompileParamsPass2();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/ELF/generated/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/ELF/generated/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace ELF
}  // namespace vpux

//
// Copyright Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace VPUIP {

//
// Passes
//

std::unique_ptr<mlir::Pass> createSetCompileParamsPass(Optional<ArchKind> arch,
                                                       Optional<CompilationMode> compilationMode,
                                                       Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createAddLinearSchedulingPass(Logger log = Logger::global());

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/VPUIP/generated/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/VPUIP/generated/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace VPUIP
}  // namespace vpux

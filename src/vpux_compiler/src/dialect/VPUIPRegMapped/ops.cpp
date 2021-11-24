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

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
// 2021_11_24: #include "vpux/compiler/dialect/VPUIPRegMapped/nce_invariant.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {}  // namespace

//
// initialize
//

void vpux::VPUIPRegMapped::VPUIPRegMappedDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUIPRegMapped/generated/ops.cpp.inc>
            >();
    /*
        addTypes<
    #define GET_TYPEDEF_LIST
    #include <vpux/compiler/dialect/VPUIPRegMapped/generated/types.cpp.inc>
                >();
    */
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIPRegMapped/generated/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIPRegMapped/generated/ops.cpp.inc>

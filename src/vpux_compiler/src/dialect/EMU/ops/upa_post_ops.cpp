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

#include "vpux/compiler/dialect/EMU/ops.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// verifyPostOp
//

mlir::LogicalResult vpux::EMU::verifyPostOp(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op != nullptr, "Got NULL pointer in verifyPostOp");
    return mlir::success();
}

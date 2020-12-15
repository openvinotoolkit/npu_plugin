//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/core/format.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(DeclareTensorOp op) {
    const auto locale = op.locale();

    if (locale == MemoryLocation::ProgrammableInput || locale == MemoryLocation::ProgrammableOutput ||
        locale == MemoryLocation::GraphFile) {
        return printTo(op.emitError(), "MemoryLocation '{0}' can't be used in '{1}'", locale,
                       DeclareTensorOp::getOperationName());
    }

    // TODO: check localeIndex

    const auto memref = op.memory().getType().cast<mlir::MemRefType>();

    if (!isMemoryCompatible(locale, memref)) {
        return printTo(op.emitError(), "'{0}' locale '{1}' is not compatible with memory space '{2}'",
                       DeclareTensorOp::getOperationName(), locale, memref.getMemorySpace());
    }

    // TODO: check other offsets

    return mlir::success();
}

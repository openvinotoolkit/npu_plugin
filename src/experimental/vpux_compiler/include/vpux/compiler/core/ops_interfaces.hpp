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

#pragma once

#include "vpux/compiler/core/dims_order.hpp"

#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>

namespace vpux {

//
// DataInfoInterface
//

class DataInfoInterface;

namespace details {

mlir::LogicalResult verifyDataInfo(mlir::Operation* op);

}  // namespace details

//
// NetInfoInterface
//

class NetInfoInterface;

namespace details {

mlir::LogicalResult verifyNetInfo(mlir::Operation* op);

mlir::FailureOr<std::pair<mlir::Operation*, mlir::FuncOp>> getNetInfo(mlir::ModuleOp module);

template <class ConcreteOp>
mlir::LogicalResult getNetInfo(mlir::ModuleOp module, ConcreteOp& netInfo, mlir::FuncOp& netFunc) {
    auto res = getNetInfo(module);
    if (mlir::failed(res)) {
        return mlir::failure();
    }

    if (!mlir::isa<ConcreteOp>(res->first)) {
        return mlir::failure();
    }

    netInfo = mlir::cast<ConcreteOp>(res->first);
    netFunc = res->second;

    return mlir::success();
}

SmallVector<DataInfoInterface, 1> getDataInfoVec(mlir::Region& region);

}  // namespace details

}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/core/generated/ops_interfaces.hpp.inc>

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

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

namespace vpux {

using CvtOpBuilderCb = FuncRef<mlir::Operation*(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Type)>;
using CvtBufferizedOpBuilderCb = FuncRef<mlir::Operation*(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>;

mlir::LogicalResult convertFunc(mlir::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                ArrayRef<mlir::Type> newResultTypes, CvtOpBuilderCb cvtOpBuilder,
                                Logger log = Logger::global());

mlir::LogicalResult convertBufferizedFunc(mlir::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                          ArrayRef<mlir::Type> newResultTypes, CvtBufferizedOpBuilderCb cvtOpBuilder,
                                          Logger log = Logger::global());

}  // namespace vpux

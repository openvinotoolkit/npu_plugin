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

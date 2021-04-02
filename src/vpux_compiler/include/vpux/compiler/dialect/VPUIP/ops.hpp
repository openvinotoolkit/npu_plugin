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

#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/structs.hpp"
#include "vpux/compiler/dialect/VPUIP/effects.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CopyOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/generated/dialect.hpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIP/generated/ops.hpp.inc>
#undef GET_OP_CLASSES

//
// Operation verifiers
//

namespace vpux {
namespace VPUIP {

constexpr Bit FP16_SIZE = 16_Bit;
constexpr KB SHAVE_LIB_DATA_SIZE = 112_KB;

mlir::LogicalResult verifyOp(DeclareTensorOp op);
mlir::LogicalResult verifyOp(DeclareConstantTensorOp op);
mlir::LogicalResult verifyOp(ConvertUPAOp op);
mlir::LogicalResult verifyOp(SoftMaxUPAOp op);
mlir::LogicalResult verifyOp(PoolingUPAOp op);
mlir::LogicalResult verifyOp(FakeQuantizeUPAOp op);
mlir::LogicalResult verifyOp(QuantCastUPAOp op);
mlir::LogicalResult verifyOp(PerAxisTileUPAOp op);
mlir::LogicalResult verifyOp(ROIPoolingUPAOp op);
mlir::LogicalResult verifyOp(NCEClusterTaskOp op);
mlir::LogicalResult verifyOp(PermuteUPAOp op);
mlir::LogicalResult verifyOp(CTCGreedyDecoderUPAOp op);
mlir::LogicalResult verifyPostOp(mlir::Operation* op);

}  // namespace VPUIP
}  // namespace vpux

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

#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/structs.hpp"
#include "vpux/compiler/dialect/VPUIP/effects.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/SCF/SCF.h>
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

//
// Operation verifiers
//

namespace vpux {
namespace VPUIP {

constexpr Bit FP16_SIZE = 16_Bit;
constexpr KB SHAVE_LIB_DATA_SIZE = 112_KB;

mlir::LogicalResult verifyOp(DeclareTensorOp op);
mlir::LogicalResult verifyOp(ConvertUPAOp op);
mlir::LogicalResult verifyOp(SoftMaxUPAOp op);
mlir::LogicalResult verifyOp(PoolingUPAOp op);
mlir::LogicalResult verifyOp(FakeQuantizeUPAOp op);
mlir::LogicalResult verifyOp(QuantCastUPAOp op);
mlir::LogicalResult verifyOp(PerAxisTileUPAOp op);
mlir::LogicalResult verifyOp(ROIPoolingUPAOp op);
mlir::LogicalResult verifyOp(ProposalUPAOp op);
mlir::LogicalResult verifyOp(PermuteUPAOp op);
mlir::LogicalResult verifyOp(CTCGreedyDecoderUPAOp op);
mlir::LogicalResult verifyOp(PadUPAOp op);
mlir::LogicalResult verifyOp(ConvolutionUPAOp op);
mlir::LogicalResult verifyOp(NCEClusterTaskOp op);
mlir::LogicalResult verifyOp(DPUTaskOp op);
mlir::LogicalResult verifyPostOp(mlir::Operation* op);

}  // namespace VPUIP
}  // namespace vpux

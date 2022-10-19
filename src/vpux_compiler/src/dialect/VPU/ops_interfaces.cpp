//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"

#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

//
// LayerOpInterface
//

mlir::LogicalResult vpux::VPU::verifyLayer(mlir::Operation* op) {
    if (op->getOperands().empty()) {
        return errorAt(op, "Layer Operation has no operands");
    }
    if (op->getResults().empty()) {
        return errorAt(op, "Layer Operation has no results");
    }

    const auto verifyType = [&](mlir::Type type, StringRef name, unsigned ind) {
        if (type.isa<mlir::MemRefType>()) {
            return errorAt(op, "Layer Operation has MemRef {0} #{1}", name, ind);
        }

        if (auto mainType = type.dyn_cast<vpux::NDTypeInterface>()) {
            if (validateQuantElemType(op->getLoc(), mainType).failed()) {
                return mlir::failure();
            }
        }

        return mlir::success();
    };

    for (auto& arg : op->getOpOperands()) {
        if (verifyType(arg.get().getType(), "operand", arg.getOperandNumber()).failed()) {
            return mlir::failure();
        }
    }
    for (auto res : op->getOpResults()) {
        if (verifyType(res.getType(), "result", res.getResultNumber()).failed()) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

//
// SparseOpInterface
//

bool vpux::VPU::supportsSparseInputs(mlir::Operation* op) {
    auto sparseOp = mlir::dyn_cast<VPU::SparseOpInterface>(op);
    if (sparseOp != nullptr) {
        return (sparseOp.sparsitySupport() & VPU::SparsitySupport::SPARSE_INPUTS) != VPU::SparsitySupport::NONE;
    }
    return false;
}

bool vpux::VPU::supportsSparseOutputs(mlir::Operation* op) {
    auto sparseOp = mlir::dyn_cast<VPU::SparseOpInterface>(op);
    if (sparseOp != nullptr) {
        return (sparseOp.sparsitySupport() & VPU::SparsitySupport::SPARSE_OUTPUTS) != VPU::SparsitySupport::NONE;
    }
    return false;
}

bool vpux::VPU::supportsSparseData(mlir::Operation* op) {
    return supportsSparseInputs(op) && supportsSparseOutputs(op);
}

//
// NCEOpInterface
//

mlir::LogicalResult vpux::VPU::details::validatePrecisionForNCE(mlir::Operation* op) {
    const auto arch = getArch(op);

    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };

    if (!NCEInvariant::isPrecisionSupported(arch, op->getOperands(), logCb)) {
        return mlir::failure();
    }
    if (!NCEInvariant::isPrecisionSupported(arch, op->getResults(), logCb)) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::details::validateWorkloadsRegion(mlir::Location loc, mlir::Region& workloads) {
    for (auto& workloadOp : workloads.getOps()) {
        if (!mlir::isa<DPUWorkloadOp>(workloadOp)) {
            return errorAt(loc, "Got unsupported Operation '{0}' in 'workloads' region", workloadOp.getName());
        }
    }

    return mlir::success();
}

mlir::Operation* vpux::VPU::details::addWorkload(mlir::Region& workloads, mlir::OpBuilder& builder, mlir::Location loc,
                                                 ShapeRef offsets, ShapeRef sizes, PaddingAttr pad, MPEMode mpeMode,
                                                 mlir::IntegerAttr clusterId) {
    if (workloads.empty()) {
        workloads.emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&workloads.front());

    const auto offsetsAttr = getIntArrayAttr(builder.getContext(), offsets);
    const auto sizesAttr = getIntArrayAttr(builder.getContext(), sizes);

    return builder.create<DPUWorkloadOp>(loc, offsetsAttr, sizesAttr, pad, mpeMode, clusterId);
}

//
// TilingBuilderOpInterface
//

mlir::Value vpux::VPU::makeTile(mlir::OpBuilder& builder, mlir::Location baseLoc, mlir::Value origVal,
                                const TileInfo& tile, StringRef valName) {
    if (tile.shape == getShape(origVal)) {
        return origVal;
    }

    const auto loc = appendLoc(baseLoc, "{0} tile {1}", valName, tile.offsets);

    auto sliceOp = builder.create<VPU::SliceOp>(loc, origVal, tile.offsets, tile.shape);
    return sliceOp.result();
}

//
// TilingInfoOpInterface
//

mlir::LogicalResult vpux::VPU::verifyTilingInfo(mlir::Operation* op) {
    if (!mlir::isa<VPU::TilingBuilderOpInterface>(op)) {
        return errorAt(op, "Operation '{0}' provides TilingInfoOpInterface, but not TilingBuilderOpInterface",
                       op->getName());
    }

    if (op->getNumResults() != 1) {
        return errorAt(op, "Unsupported operation '{0}', it must have one and only one result", op->getName());
    }

    return mlir::success();
}

//
// EltwiseOp
//

mlir::LogicalResult vpux::VPU::verifyEltwiseOp(mlir::Operation* op) {
    if (!mlir::isa<VPU::LayerOpInterface>(op)) {
        return errorAt(op, "EltwiseOp trait is applied to non layer operation");
    }

    if (op->getNumResults() != 1) {
        return errorAt(op, "Operation with multiple results can't be EltwiseOp");
    }

    const auto outputShape = op->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();
    if (llvm::none_of(op->getOperands(), [&](mlir::Value operand) {
            return operand.getType().cast<vpux::NDTypeInterface>().getShape() == outputShape;
        })) {
        return errorAt(op, "EltwiseOp must have at least one input shape equal to the output shape");
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU/generated/ops_interfaces.cpp.inc>

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"

#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

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

    const auto logCb = [op](const llvm::formatv_object_base& msg) {
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
// Generated
//

#include <vpux/compiler/dialect/VPU/generated/ops_interfaces.cpp.inc>

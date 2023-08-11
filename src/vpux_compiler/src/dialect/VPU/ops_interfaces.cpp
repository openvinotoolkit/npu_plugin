//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

//
// SparseOpInterface
//

bool vpux::VPU::supportsSparseInputs(mlir::Operation* op) {
    const auto compressedInput = [](mlir::Value operand) {
        auto inputShape = operand.getType().cast<vpux::NDTypeInterface>().getShape();
        if (inputShape[Dims4D::Act::C] < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT) {
            return true;
        }
        return false;
    };
    if (compressedInput(op->getOperand(0))) {
        return false;
    }
    if (mlir::isa<VPU::NCEEltwiseOp>(op) && compressedInput(op->getOperand(1))) {
        return false;
    }

    if (auto sparseOp = mlir::dyn_cast<VPU::SparseOpInterface>(op)) {
        return VPU::bitEnumContains(sparseOp.sparsitySupport(), VPU::SparsitySupport::SPARSE_INPUTS);
    }
    return false;
}

bool vpux::VPU::supportsSparseOutputs(mlir::Operation* op) {
    if (auto sparseOp = mlir::dyn_cast<VPU::SparseOpInterface>(op)) {
        return VPU::bitEnumContains(sparseOp.sparsitySupport(), VPU::SparsitySupport::SPARSE_OUTPUTS);
    }
    return false;
}

bool vpux::VPU::supportsSparseWeights(mlir::Operation* op) {
    if (auto sparseOp = mlir::dyn_cast<VPU::SparseOpInterface>(op)) {
        return VPU::bitEnumContains(sparseOp.sparsitySupport(), VPU::SparsitySupport::SPARSE_WEIGHTS);
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

    if (op->hasAttr("auto_broadcast")) {
        auto autoBroadcast = op->getAttr("auto_broadcast").dyn_cast<IE::AutoBroadcastTypeAttr>();
        if (autoBroadcast == nullptr) {
            return errorAt(op, "Auto broadcast attribute cannot be cast");
        }
        auto broadcast = autoBroadcast.getValue();

        SmallVector<ArrayRef<int64_t>> inputShapes;
        for (auto operand : op->getOperands()) {
            const auto shape = operand.getType().cast<vpux::NDTypeInterface>().getShape().raw();
            inputShapes.push_back(shape);
        }

        const auto outputShape = IE::broadcastEltwiseShape(inputShapes, broadcast, op->getLoc());

        if (mlir::failed(outputShape)) {
            return errorAt(op, "Eltwise inputs cannot be broadcast");
        }
    }

    return mlir::success();
}

//
// NCEOpInterface
//

mlir::LogicalResult vpux::VPU::verifyNCEOp(mlir::Operation* op) {
    if (!mlir::isa<VPU::NCEOpInterface>(op)) {
        return errorAt(op, "Operation '{0}' is not NCE", op->getName());
    }

    auto nceOp = mlir::cast<VPU::NCEOpInterface>(op);
    if (vpux::VPU::details::validatePrecisionForNCE(nceOp).failed()) {
        return mlir::failure();
    }
    if (vpux::VPU::details::validateWorkloadsRegion(nceOp->getLoc(), nceOp.workloads()).failed()) {
        return mlir::failure();
    }

    if (auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        return iface.verifyChannels();
    }

    return mlir::success();
}

//
// SameInOutDefaultDimsOrder
//

mlir::LogicalResult vpux::VPU::verifySameInOutDefaultDimsOrder(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layer != nullptr, "Operation {0} does not implement VPU::LayerOpInterface", op->getName());

    const auto input = layer.getInputs()[0];
    const auto output = layer.getOutputs()[0];

    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromValue(output);

    if (inOrder != outOrder) {
        return errorAt(op->getLoc(), "Operation must have the same input and output order. inL={0}, outL={1}", inOrder,
                       outOrder);
    }

    return mlir::success();
}

void vpux::VPU::inferLayoutInfoSameInOutDefaultDimsOrder(IE::LayerLayoutInfo& info) {
    const auto filter = [](size_t ind) {
        return ind != 0;
    };
    IE::fillDefaultLayoutInfo(info, filter, filter);

    info.setOutput(0, info.getInput(0));
}

//
// SameAnyDimsOrder
//

mlir::LogicalResult vpux::VPU::verifySameAnyDimsOrder(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    if (layer == nullptr) {
        return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
    }

    auto inputs = layer.getInputs();

    const auto firstInput = inputs.front();
    const auto mainOrder = DimsOrder::fromValue(firstInput);

    for (const auto& val : layer->getOpOperands()) {
        const auto order = DimsOrder::fromValue(val.get());

        if (order != mainOrder) {
            return errorAt(op, "Operation's input/output layout mismatch");
        }
    }

    return mlir::success();
}

void vpux::VPU::inferLayoutInfoSameAnyDimsOrder(IE::LayerLayoutInfo& info) {
    const auto inOrder = info.getInput(0);
    info.fill(inOrder);
}

//
// SameInOutSpecificDimsOrder
//

mlir::LogicalResult vpux::VPU::verifySameInOutSpecificDimsOrder(mlir::Operation* op,
                                                                ArrayRef<DimsOrder> supportedLayouts) {
    if (verifySameInOutDefaultDimsOrder(op).failed()) {
        return mlir::failure();
    }

    auto layerOp = mlir::dyn_cast<VPU::LayerOpInterface>(op);
    VPUX_THROW_UNLESS(layerOp != nullptr, "Operation {0} does not implement VPU::LayerOpInterface", op->getName());

    const auto input = layerOp.getInputs()[0];
    const auto inOrder = DimsOrder::fromValue(input);

    const auto isSupported = std::count(supportedLayouts.begin(), supportedLayouts.end(), inOrder);
    if (!isSupported) {
        return errorAt(op->getLoc(), "Operation does not support {0} layout", inOrder);
    }

    return mlir::success();
}

void vpux::VPU::inferLayoutInfoSameInOutSpecificDimsOrder(IE::LayerLayoutInfo& info,
                                                          ArrayRef<DimsOrder> supportedLayouts) {
    const auto filter = [](size_t ind) {
        return ind != 0;
    };
    IE::fillDefaultLayoutInfo(info, filter, filter);

    const auto mainOrder = info.getInput(0);

    if (llvm::is_contained(supportedLayouts, mainOrder)) {
        info.setOutput(0, mainOrder);
        return;
    }

    const auto supportedOrderIt = llvm::find_if(supportedLayouts, [mainOrder](DimsOrder order) {
        return order.numDims() == mainOrder.numDims();
    });

    VPUX_THROW_UNLESS(supportedOrderIt != supportedLayouts.end(),
                      "Layouts supported by the operation '{0}' do not match the rank '{1}' of the input shape",
                      supportedLayouts, mainOrder.numDims());

    const auto supportedOrder = *supportedOrderIt;
    info.setInput(0, supportedOrder);
    info.setOutput(0, supportedOrder);
}

//
// isPureViewLike
//

bool vpux::VPU::isPureViewOp(mlir::Operation* op) {
    return mlir::isa<VPU::ViewLikeOpInterface, mlir::ViewLikeOpInterface, vpux::MultiViewOpInterface,
                     vpux::GroupedViewOpInterface>(op);
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU/generated/ops_interfaces.cpp.inc>

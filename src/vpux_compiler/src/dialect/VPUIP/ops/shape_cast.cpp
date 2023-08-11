//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::Value VPUIP::ShapeCastOp::getViewSource() {
    return source();
}

mlir::LogicalResult vpux::VPUIP::ShapeCastOp::verify() {
    const auto op = getOperation();
    const auto inType = source().getType().cast<vpux::NDTypeInterface>();
    const auto outType = result().getType().cast<vpux::NDTypeInterface>();

    if (inType.getDimsOrder() != outType.getDimsOrder()) {
        return errorAt(op, "Input dims order '{0}' doesn't match output dims order '{1}'", inType.getDimsOrder(),
                       outType.getDimsOrder());
    }
    if (inType.getRank() != outType.getRank()) {
        return errorAt(op, "Input rank '{0}' doesn't match output rank '{1}'", inType.getRank(), outType.getRank());
    }
    if (inType.getElementType() != outType.getElementType()) {
        return errorAt(op, "Input element type '{0}' doesn't match output element type '{1}'", inType.getElementType(),
                       outType.getElementType());
    }
    if (inType.getMemSpace() != outType.getMemSpace()) {
        return errorAt(op, "Input mem space '{0}' doesn't match output mem space '{1}'", inType.getMemSpace(),
                       outType.getMemSpace());
    }

    return mlir::success();
}

vpux::NDTypeInterface checkAndUpdateDistributedType(vpux::VPUIP::DistributedBufferType inTypeDistr,
                                                    llvm::SmallVector<int64_t> shape, VPU::ArchKind arch) {
    const auto distrMode = inTypeDistr.getDistribution().mode().getValue();
    if (VPU::bitEnumContains(distrMode, VPU::DistributionMode::SEGMENTED) ||
        VPU::bitEnumContains(distrMode, VPU::DistributionMode::OVERLAPPED)) {
        const auto origShape = inTypeDistr.getShape().raw();
        const auto isSameDimAsClustering = [&]() {
            const auto numTiles = parseIntArrayAttr<int64_t>(inTypeDistr.getDistribution().num_tiles());
            for (auto dim : irange(origShape.size())) {
                if (numTiles[dim] > 1 && origShape[dim] != shape[dim]) {
                    return false;
                }
            }
            return true;
        };
        VPUX_THROW_UNLESS(isSameDimAsClustering() ||
                                  VPUIP::isDistributedCompatibleAfterShapeChange(inTypeDistr, ShapeRef(shape), arch),
                          "Cannot cast shape from '{0}' to '{1}' as clustering", origShape, shape);
    }

    auto outType = inTypeDistr.cast<NDTypeInterface>().changeShape(ShapeRef(shape));
    const auto order = outType.getDimsOrder();
    const auto ctx = inTypeDistr.getContext();
    const auto mode = inTypeDistr.getDistribution().mode().getValue();
    const auto newDistribution =
            mode == VPU::DistributionMode::SEGMENTED && VPUIP::isSegmentedOverH(inTypeDistr.getDistribution())
                    ? getSOHDistAttrWithNewShape(ctx, inTypeDistr, ShapeRef(shape), arch)
                    : inTypeDistr.getDistribution();
    outType = VPUIP::DistributedBufferType::get(ctx, shape, outType.getElementType(),
                                                mlir::AffineMapAttr::get(order.toAffineMap(ctx)), outType.getMemSpace(),
                                                newDistribution);

    if (!inTypeDistr.getLayout().isa<mlir::AffineMapAttr>()) {
        const auto memStrides = StrideReqs::compact(order.numDims()).calcStrides(order, outType);
        auto compactStrides = order.toLogicalOrder(memStrides);
        auto newOutType = outType.changeStrides(StridesRef(compactStrides));
        return newOutType;
    }

    return outType;
}

//
// InferTypeOpInterface
//

mlir::LogicalResult VPUIP::ShapeCastOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::ShapeCastOpAdaptor shapeCast(operands, attrs);
    if (mlir::failed(shapeCast.verify(loc))) {
        return mlir::failure();
    }
    const auto arch = VPU::getArch(operands[0].isa<mlir::BlockArgument>()
                                           ? operands[0].getParentRegion()->getParentOfType<mlir::ModuleOp>()
                                           : operands[0].getDefiningOp());
    const auto input = shapeCast.source();
    const auto inType = input.getType();
    const auto shape = parseIntArrayAttr<int64_t>(shapeCast.shape());

    if (auto inTypeDistr = inType.dyn_cast<VPUIP::DistributedBufferType>()) {
        inferredReturnTypes.push_back(checkAndUpdateDistributedType(inTypeDistr, shape, arch));
        return mlir::success();
    } else if (auto inTypeSparse = inType.dyn_cast<VPUIP::SparseBufferType>()) {
        if (auto data = inTypeSparse.getData().dyn_cast<VPUIP::DistributedBufferType>()) {
            const auto newData = checkAndUpdateDistributedType(data, shape, arch);
            if (data.getLayout().isa<mlir::AffineMapAttr>()) {
                inferredReturnTypes.push_back(inTypeSparse.changeShape(ShapeRef(shape)));
            } else {
                inferredReturnTypes.push_back(
                        inTypeSparse.changeShape(ShapeRef(shape)).changeStrides(newData.getStrides()));
            }
            return mlir::success();
        }
    }

    const auto inTypeND = input.getType().cast<NDTypeInterface>();
    const auto outType = inTypeND.changeShape(ShapeRef(shape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

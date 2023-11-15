//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/attributes.hpp"
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

namespace {
VPU::DistributedTensorAttr getDistributedAttrAfterShapeCast(VPUIP::DistributedBufferType origDistrType,
                                                            ArrayRef<int64_t> shape, VPU::ArchKind arch) {
    const auto origDistribution = origDistrType.getDistribution();
    const auto mode = origDistribution.getMode().getValue();
    const auto origShape = origDistrType.getShape().raw();
    auto ctx = origDistrType.getContext();

    const auto isSameDimAsClustering = [&]() {
        const auto numTiles = parseIntArrayAttr<int64_t>(origDistribution.getNumTiles());
        for (auto dim : irange(origShape.size())) {
            if (numTiles[dim] > 1 && origShape[dim] != shape[dim]) {
                return true;
            }
        }
        return false;
    };

    // ShapeCastOp is not a "compute" op, therefore memory and compute shapes are the same
    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(origDistribution)) {
        VPUX_THROW_WHEN((mode != VPU::DistributionMode::OVERLAPPED) && (mode != VPU::DistributionMode::SEGMENTED) &&
                                !VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED) &&
                                !VPU::bitEnumContains(mode, VPU::DistributionMode::MULTICASTED),
                        "Cannot cast shape with explicit memory/compute shapes/offsets with DistributionMode {0}",
                        origDistribution.getMode());

        const auto numClusters = static_cast<size_t>(origDistribution.getNumClusters().getInt());

        // When having full output broadcasted across all clusters, offsets stay the same as original ones
        // and shape changes to the new one indicated by ShapeCast
        const auto shapeVec = SmallVector<int64_t>(shape.begin(), shape.end());
        auto perClusterShapes = SmallVector<SmallVector<int64_t>>(numClusters, shapeVec);

        // SEGMENTED/OVERLAPPED case
        if ((mode == VPU::DistributionMode::OVERLAPPED) || (mode == VPU::DistributionMode::SEGMENTED)) {
            VPUX_THROW_WHEN(
                    isSameDimAsClustering(),
                    "Cannot cast shape from '{0}' to '{1}' when having explicit memory/compute shapes/offsets as "
                    "segmentation dim changes at output",
                    origShape, shape);

            // Use newly casted shape for all dims except the clustering dim
            const auto origPerClusterShapes = parseIntArrayOfArrayAttr<int64_t>(origDistribution.getMemoryShapes());
            const auto numTiles = parseIntArrayAttr<int64_t>(origDistribution.getNumTiles());
            for (size_t cluster = 0; cluster < numClusters; cluster++) {
                for (size_t dim = 0; dim < shape.size(); dim++) {
                    if (numTiles[dim] != 1) {
                        perClusterShapes[cluster][dim] = origPerClusterShapes[cluster][dim];
                    }
                }
            }
        }

        auto perClusterShapesAttr = vpux::getIntArrayOfArray(ctx, perClusterShapes);
        return VPU::DistributedTensorAttr::get(
                ctx, origDistribution.getMode(), origDistribution.getNumTiles(), origDistribution.getKernel(),
                origDistribution.getPads(), origDistribution.getStrides(), origDistribution.getNumClusters(),
                origDistribution.getAlignment(), origDistribution.getUniformDistributedSegments(), perClusterShapesAttr,
                origDistribution.getMemoryOffsets(), perClusterShapesAttr, origDistribution.getMemoryOffsets(),
                origDistribution.getEqualMemoryAndComputeView());
    }

    if (VPU::bitEnumContains(mode, VPU::DistributionMode::SEGMENTED)) {
        VPUX_THROW_WHEN(isSameDimAsClustering() &&
                                !VPUIP::isDistributedCompatibleAfterShapeChange(origDistrType, ShapeRef(shape), arch),
                        "Cannot cast shape from '{0}' to '{1}' as clustering", origShape, shape);
    }

    VPUX_THROW_WHEN((mode == VPU::DistributionMode::OVERLAPPED) && isSameDimAsClustering(),
                    "Cannot cast shape from '{0}' to '{1}' as OVERLAPPED clustering; clustering dim changes at output",
                    origShape, shape);

    if (VPU::isSegmentedOverH(origDistribution)) {
        return getSOHDistAttrWithNewShape(ctx, origDistrType, ShapeRef(shape), arch);
    }

    return origDistribution;
}
}  // namespace

vpux::NDTypeInterface checkAndUpdateDistributedType(vpux::VPUIP::DistributedBufferType inTypeDistr,
                                                    ArrayRef<int64_t> shape, VPU::ArchKind arch) {
    const auto ctx = inTypeDistr.getContext();
    auto newDistribution = getDistributedAttrAfterShapeCast(inTypeDistr, shape, arch);
    auto outType = inTypeDistr.changeShapeForExplicitDistribution(ShapeRef(shape), newDistribution);

    return VPUIP::DistributedBufferType::get(ctx, shape, outType.getElementType(),
                                             mlir::AffineMapAttr::get(outType.getDimsOrder().toAffineMap(ctx)),
                                             outType.getMemSpace(), newDistribution);
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

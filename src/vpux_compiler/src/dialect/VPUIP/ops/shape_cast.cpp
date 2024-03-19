//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/strides_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/reshape_utils.hpp"

using namespace vpux;

mlir::Value VPUIP::ShapeCastOp::getViewSource() {
    return getSource();
}

mlir::LogicalResult vpux::VPUIP::ShapeCastOp::verify() {
    const auto op = getOperation();
    const auto inType = getSource().getType().cast<vpux::NDTypeInterface>();
    const auto outType = getResult().getType().cast<vpux::NDTypeInterface>();

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
VPU::DistributedTensorAttr getDistributedAttrAfterShapeCast(VPU::DistributedTypeInterface origDistrType,
                                                            ArrayRef<int64_t> shape, VPU::ArchKind arch) {
    auto distributedBuffer = origDistrType.getDistributedTypes().front().cast<VPUIP::DistributedBufferType>();
    auto origDistribution = distributedBuffer.getDistribution();
    const auto mode = origDistribution.getMode().getValue();

    auto ndTypeIf = origDistrType.cast<NDTypeInterface>();
    const auto origShape = ndTypeIf.getShape().raw();
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
                                !VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) &&
                                !VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED),
                        "Cannot cast shape with explicit memory/compute shapes/offsets with DistributionMode {0}",
                        origDistribution.getMode());

        if (auto sparseBuff = origDistrType.dyn_cast<VPUIP::SparseBufferType>()) {
            origDistribution = VPU::getExplicitDistrAttrForActualDataFromSparseType(sparseBuff);
        }

        // When having input broadcasted across all clusters, ShapeCast can set its output as DUPLICATED,
        // regardless of input mode. The reason for that is because ShapeCast is not a compute op and
        // therefore compute shapes/offsets mean nothing to it. Moreover, in cases such as SEG|DUP where
        // a tile axis or alignment exists, the ShapeCast's new shape might not be compatible with those
        // attributes anymore so it would be better to discard them.
        if (VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
            VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED)) {
            auto duplicatedOutputMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
            return VPU::getNonOverlappedDistributedAttr(Shape(shape), duplicatedOutputMode, nullptr,
                                                        origDistribution.getNumClusters(), nullptr,
                                                        origDistribution.getUniformDistributedSegments(), ctx);
        }

        const auto numClusters = static_cast<size_t>(origDistribution.getNumClusters().getInt());

        const auto shapeVec = SmallVector<int64_t>(shape.begin(), shape.end());
        auto perClusterShapes = SmallVector<SmallVector<int64_t>>(numClusters, shapeVec);

        const bool clusteringDimChanges = isSameDimAsClustering();
        if (VPU::isSegmentedOverH(origDistribution)) {
            VPUX_THROW_WHEN(
                    clusteringDimChanges && !VPUIP::isDistributedCompatibleAfterShapeChangeForViewOps(
                                                    distributedBuffer, ShapeRef(shape), ndTypeIf.getDimsOrder(), arch),
                    "Cannot cast shape from '{0}' to '{1}' as clustering", origShape, shape);

            return getSOHDistAttrWithNewShape(ctx, distributedBuffer, ShapeRef(shape), arch);
        }

        // SEGMENTED/OVERLAPPED case
        if ((mode == VPU::DistributionMode::OVERLAPPED) || (mode == VPU::DistributionMode::SEGMENTED)) {
            VPUX_THROW_WHEN(clusteringDimChanges,
                            "Cannot cast shape from '{0}' to '{1}' when having explicit memory/compute "
                            "shapes/offsets as segmentation dim changes at output",
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

    auto dataShape = shape;
    if (auto sparseBuff = origDistrType.dyn_cast<VPUIP::SparseBufferType>()) {
        if (auto seAttr = sparseBuff.getSeAttr()) {
            dataShape = seAttr.backInferInputShape(ShapeRef(shape)).raw();
        }
    }

    if (VPU::bitEnumContainsAny(mode, VPU::DistributionMode::SEGMENTED)) {
        VPUX_THROW_WHEN(isSameDimAsClustering() &&
                                !VPUIP::isDistributedCompatibleAfterShapeChangeForViewOps(
                                        distributedBuffer, ShapeRef(dataShape), ndTypeIf.getDimsOrder(), arch),
                        "Cannot cast shape from '{0}' to '{1}' as clustering", origShape, shape);
    }

    VPUX_THROW_WHEN((mode == VPU::DistributionMode::OVERLAPPED) && isSameDimAsClustering(),
                    "Cannot cast shape from '{0}' to '{1}' as OVERLAPPED clustering; clustering dim changes at output",
                    origShape, shape);

    if (VPU::isSegmentedOverH(origDistribution)) {
        return getSOHDistAttrWithNewShape(ctx, distributedBuffer, ShapeRef(dataShape), arch);
    }

    return origDistribution;
}
}  // namespace

vpux::NDTypeInterface checkAndUpdateDistributedType(VPU::DistributedTypeInterface inTypeDistr, ArrayRef<int64_t> shape,
                                                    VPU::ArchKind arch) {
    const auto ctx = inTypeDistr.getContext();
    auto newDistribution = getDistributedAttrAfterShapeCast(inTypeDistr, shape, arch);
    auto outType = inTypeDistr.changeShapeForExplicitDistribution(ShapeRef(shape), newDistribution);

    return VPUIP::DistributedBufferType::get(ctx, shape, outType.getElementType(),
                                             mlir::AffineMapAttr::get(outType.getDimsOrder().toAffineMap(ctx)),
                                             outType.getMemSpace(), newDistribution);
}

// If the ShapeCast input type has strides attribution, the output should infer a strides
// to ensure it has the same buffer distribution. Otherwise it will has accuracy issue.
// There is no guarantee that it will always get a legal output strides.
// If return 'std::nullopt' that mean this ShapeCast Op is illegal.
// Generally, the legal ShapeCast op has the following characteristics:
// 1. The input stride only exist on one axis;
// 2. The 'inStridesDim' should be on one axis or split into continuous axes on the output.
//    It means Stride Dim can not mixed with std::nullopt Stride Dims.
// 3. Split by 'stridesDim', input and output memory shape can be divided into three parts:
//    [inputLeftDimTotalSize,  inputStridesDimSize,  inputRightDimTotalSize] should equal with
//    [outputLeftDimTotalSize, outputStridesDimSize, outputRightDimTotalSize]
std::optional<Strides> inferShapeCastOutputStrides(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) {
    VPUX_THROW_UNLESS(inType.getRank() == outType.getRank(),
                      "Input type '{0}' and Output type '{1}' has different tensor rank", inType, outType);

    if (inType.getShape().totalSize() != outType.getShape().totalSize()) {
        return outType.getStrides();
    }

    const auto inStridesMemDims = VPUIP::getStridesMemDims(inType);
    // Limitation 1: Stride only exist in one axis
    // - Legal case:   memShape: 1x32x512x16, memStrides: [524288, 16384, 32, 1]
    //   The stridesMemDim is Dims4D::Act::H
    // - Illegal case: memShape: 1x32x512x16, memStrides: [1048576, 32768, 32, 1]
    //   The stridesMemDim are Dims4D::Act::C and Dims4D::Act::H
    if (inStridesMemDims.size() > 1) {
        return std::nullopt;
    }

    if (inStridesMemDims.empty()) {
        return outType.getStrides();
    }

    // Limitation 2&3: The 'inStridesDim' should be on one axis or split into continuous axes on the output
    // - Legal case 1: inMemShape: 1x32x512x16, inMemStrides: [524288, 16384, 32, 1], outMemShape: 2x16x512x16
    //   The outStridesDims is Dims4D::Act::H and [1x32, 512, 16] == [2x16, 512, 16]
    // - Legal case 2: inMemShape: 1x1x256x512, inMemStrides: [262144, 262144, 1024, 1], outMemShape: 1x16x16x512
    //   The outStridesDims is [Dims4D::Act::C, Dims4D::Act::H] and [1x1, 256, 512] = [1, 16x16, 512]
    // - Illegal case: inMemShape: 1x32x512x16, inMemStrides: [524288, 16384, 32, 1], outMemShape: 4x16x512x8
    //   The stridesMemDim is Dims4D::Act::H, but [1x32, 512, 16] != [4x16, 512, 8]
    const auto inMemShape = inType.getMemShape();
    const auto outMemShape = outType.getMemShape();
    const auto inStridesMemDim = inStridesMemDims.front();
    const auto legalOutputStridesDims = vpux::deduceLegalOutputMemDims(inMemShape, outMemShape, inStridesMemDim);
    if (!legalOutputStridesDims.has_value()) {
        return std::nullopt;
    }
    const auto outStridesDims = legalOutputStridesDims.value();

    const auto outOrder = outType.getDimsOrder();
    const auto outElemSize = outType.getElemTypeSize();
    auto outMemStrides = StrideReqs::compact(outOrder.numDims()).calcStrides(outElemSize, outMemShape);

    const auto inMemStrides = inType.getMemStrides();
    const auto outStridesDimLeftBoundary = outStridesDims.back();
    outMemStrides[outStridesDimLeftBoundary] = inMemStrides[inStridesMemDim];
    for (auto ind = outStridesDimLeftBoundary.ind() - 1; ind >= 0; ind--) {
        const auto currentMemDim = MemDim(ind);
        const auto prevMemDim = MemDim(ind + 1);
        outMemStrides[currentMemDim] = outMemStrides[prevMemDim] * outMemShape[prevMemDim];
    }

    return outOrder.toLogicalOrder(outMemStrides);
}

//
// InferTypeOpInterface
//

mlir::LogicalResult VPUIP::ShapeCastOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::ShapeCastOpAdaptor shapeCast(operands, attrs);
    if (mlir::failed(shapeCast.verify(loc))) {
        return mlir::failure();
    }
    const auto arch = VPU::getArch(operands[0].isa<mlir::BlockArgument>()
                                           ? operands[0].getParentRegion()->getParentOfType<mlir::ModuleOp>()
                                           : operands[0].getDefiningOp());
    const auto inType = shapeCast.getSource().getType().cast<NDTypeInterface>();
    const auto outShape = parseIntArrayAttr<int64_t>(shapeCast.getShape());

    vpux::NDTypeInterface outType;
    auto inTypeDistr = inType.dyn_cast<VPU::DistributedTypeInterface>();
    if (inTypeDistr != nullptr && inTypeDistr.containsDistributedTypes()) {
        outType = checkAndUpdateDistributedType(inTypeDistr, outShape, arch);
    } else {
        outType = inType.changeShape(ShapeRef(outShape));
    }

    const auto outputStrides = inferShapeCastOutputStrides(inType, outType);
    if (!outputStrides.has_value()) {
        return mlir::failure();
    }
    const auto outputStridesVal = outputStrides.value();

    inferredReturnTypes.push_back(outType.getStrides() != outputStridesVal ? outType.changeStrides(outputStridesVal)
                                                                           : outType);

    return mlir::success();
}

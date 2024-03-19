//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/reshape_utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

using GetCopyFunctType = FuncRef<VPUIP::LayerOpInterface(mlir::Operation*)>;
using CreateCopyFunctType =
        FuncRef<VPUIP::LayerOpInterface(mlir::PatternRewriter&, mlir::Location, mlir::Value, mlir::Value)>;

VPUIP::LayerOpInterface getCopyOp(mlir::Operation* sourceOp) {
    return mlir::dyn_cast_or_null<VPUIP::CopyOp>(sourceOp);
}

VPUIP::LayerOpInterface createNewCopyOp(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                                        mlir::Value outputBuff) {
    return rewriter.create<VPUIP::CopyOp>(loc, input, outputBuff);
}

VPUIP::LayerOpInterface getTillingCopyOp(mlir::Operation* sourceOp) {
    auto clusterTiling = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(sourceOp);
    if (clusterTiling == nullptr || clusterTiling.getInnerTaskOpOfType<VPUIP::CopyOp>() == nullptr) {
        return nullptr;
    }

    return clusterTiling;
}

VPUIP::LayerOpInterface createNewTillingCopyOp(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                                               mlir::Value outputBuff) {
    const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
    };

    SmallVector<mlir::Value> inputsOutputOperands = {input, outputBuff};
    return rewriter.create<VPUIP::NCEClusterTilingOp>(loc, outputBuff.getType(), inputsOutputOperands,
                                                      copyOutBodyBuilder);
}

// Try to get reshape IO axes mapping when below two conditions are met:
// 1.MemShape on target axis is not changed by reshaping.
// 2.Data total size is not changed on both higher and lower dimension.

// For example: reshape 2x64x64x32 to 128x64x4x8x1 and input axis is [d2]
// We will get output axis [d1] and this function returns axis mapping {d2, d1}
//  - inMemShape[d2] = 64 and
//    outMemShape[d1] = 64
//  - Input DataTotalSize on d2 higher dimension is 128 (2x64) and
//    output DataTotalSize on d1 higher dimension is 128
//  - Input DataTotalSize on d2 lower dimension is 32 and
//    output DataTotalSize on d1 higher dimension is 32 (4x8x1)

// This function would reture mlir::failure() if can not find IO axes mapping successfully.
// Return {-1, -1} to indicate there's no numTiles and alignment attributes in distribution.
mlir::FailureOr<std::pair<int64_t, int64_t>> getDistributedAxesMappingAfterShapeChanged(
        vpux::NDTypeInterface reshapeInType, vpux::NDTypeInterface reshapeOutType,
        VPU::DistributedTensorAttr copyInDistribution, Logger log) {
    auto inOrder = reshapeInType.getDimsOrder();
    auto outOrder = reshapeOutType.getDimsOrder();

    auto parseMaxElemIndexFromArray = [](ArrayRef<int64_t> array) -> mlir::FailureOr<int64_t> {
        const auto numDimsGreaterThanOne = std::count_if(array.begin(), array.end(), [](int64_t v) {
            return v > 1;
        });
        if (numDimsGreaterThanOne != 1) {
            return mlir::failure();
        }

        auto maxElem = std::max_element(array.begin(), array.end());
        return std::distance(array.begin(), maxElem);
    };

    int64_t numTilesAxis = -1;
    int64_t alignmentAxis = -1;

    auto numTilesAttr = copyInDistribution.getNumTiles();
    if (numTilesAttr != nullptr) {
        const auto numTilesVec = parseIntArrayAttr<int64_t>(numTilesAttr);
        auto parseNumTilesAxis = parseMaxElemIndexFromArray(numTilesVec);
        if (!mlir::failed(parseNumTilesAxis)) {
            numTilesAxis = parseNumTilesAxis.value();
        }
    }

    auto alignmentAttr = copyInDistribution.getAlignment();
    if (alignmentAttr != nullptr) {
        const auto alignmentVec = parseIntArrayAttr<int64_t>(alignmentAttr);
        auto parseAlignmentAxis = parseMaxElemIndexFromArray(alignmentVec);
        if (!mlir::failed(parseAlignmentAxis)) {
            alignmentAxis = parseAlignmentAxis.value();
        }
    }

    if (numTilesAxis != -1 && alignmentAxis != -1 && numTilesAxis != alignmentAxis) {
        log.trace("Unexpected numTilesAxis {0} and alignmentAxis {1} in distribution {2}", numTilesAxis, alignmentAxis,
                  copyInDistribution);
        return mlir::failure();
    }

    auto inAxis = numTilesAxis;
    if (numTilesAxis == -1) {
        inAxis = alignmentAxis;
    }

    if (inAxis == -1) {
        log.trace("Distribution {0} does not contain numTiles or alignment attribute", copyInDistribution);
        return std::pair(numTilesAxis, alignmentAxis);
    }

    auto inMemDim = inOrder.toMemDim(Dim(inAxis));
    const auto inMemShape = reshapeInType.getMemShape();
    const auto outMemShape = reshapeOutType.getMemShape();
    const auto legalOutputMemDims = vpux::deduceLegalOutputMemDims(inMemShape, outMemShape, inMemDim);
    if (!legalOutputMemDims.has_value()) {
        return mlir::failure();
    }

    auto outMemDims = legalOutputMemDims.value();
    if (outMemDims.size() != 1) {
        return mlir::failure();
    }

    int64_t outAxis = outOrder.toDim(*outMemDims.begin()).ind();

    log.trace("Got IO axes mapping {0} -> {1}", inAxis, outAxis);

    return std::make_pair(inAxis, outAxis);
}

VPU::DistributedTensorAttr changeDistributedAxisOnDistributedTensorAttr(VPU::DistributedTensorAttr inDistribution,
                                                                        int64_t inDistributionAxis,
                                                                        int64_t outDistributionAxis,
                                                                        int64_t outputRank) {
    auto ctx = inDistribution.getContext();

    auto generateNewArray = [&](ArrayRef<int64_t> srcArray, int64_t inAxis, int64_t outAxis) -> SmallVector<int64_t> {
        SmallVector<int64_t> newArray(outputRank, 1);
        VPUX_THROW_UNLESS(inAxis >= 0 && inAxis < checked_cast<int64_t>(srcArray.size()),
                          "Input axis index is out of range {0}", inAxis);
        VPUX_THROW_UNLESS(outAxis >= 0 && outAxis < outputRank, "Output axis index is out of range {0}", outAxis);
        newArray[outAxis] = srcArray[inAxis];
        return newArray;
    };

    auto numTilesAttr = inDistribution.getNumTiles();
    if (numTilesAttr != nullptr) {
        const auto numTilesVec = parseIntArrayAttr<int64_t>(numTilesAttr);
        numTilesAttr = getIntArrayAttr(ctx, generateNewArray(numTilesVec, inDistributionAxis, outDistributionAxis));
    }

    auto alignmentAttr = inDistribution.getAlignment();
    if (alignmentAttr != nullptr) {
        const auto alignmentVec = parseIntArrayAttr<int64_t>(alignmentAttr);
        alignmentAttr = getIntArrayAttr(ctx, generateNewArray(alignmentVec, inDistributionAxis, outDistributionAxis));
    }

    return VPU::DistributedTensorAttr::get(ctx, inDistribution.getMode(), numTilesAttr, inDistribution.getKernel(),
                                           inDistribution.getPads(), inDistribution.getStrides(),
                                           inDistribution.getNumClusters(), alignmentAttr,
                                           inDistribution.getUniformDistributedSegments(), nullptr, nullptr, nullptr,
                                           nullptr, inDistribution.getEqualMemoryAndComputeView());
}

//
// LayerRewriter
//

class LayerRewriterBase : public mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface> {
public:
    LayerRewriterBase(mlir::MLIRContext* ctx, GetCopyFunctType getCopyOp, CreateCopyFunctType createNewCopyOp,
                      Logger log)
            : mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface>(ctx),
              _getCopyOp(getCopyOp),
              _createNewCopyOp(createNewCopyOp),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::ViewLikeOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    GetCopyFunctType _getCopyOp;
    CreateCopyFunctType _createNewCopyOp;
    Logger _log;
};

mlir::LogicalResult LayerRewriterBase::matchAndRewrite(mlir::ViewLikeOpInterface origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    if (mlir::isa<VPUIP::LayerOpInterface>(*origOp)) {
        return mlir::failure();
    }

    if (!mlir::isa<VPUIP::PermuteCastOp, VPUIP::GenericReshapeOp, VPUIP::QuantizeCastOp, VPUIP::ShapeCastOp>(*origOp)) {
        return mlir::failure();
    }

    _log.trace("Got pure view-like op: '{0}':'{1}'", origOp->getName(), origOp->getLoc());
    auto maybeCopy = _getCopyOp(origOp->getOperand(0).getDefiningOp());
    if (maybeCopy == nullptr) {
        StringRef parentOpName = "None";
        if (auto parentOp = origOp->getOperand(0).getDefiningOp()) {
            parentOpName = parentOp->getName().getStringRef();
        }
        _log.trace("The operation defining the input is not Copy: '{0}'", parentOpName);
        return mlir::failure();
    }

    auto copyOpInput = maybeCopy.getInputs()[0];
    auto copyOpOutput = maybeCopy.getOutputs()[0];
    // When we have compress convolution we don't want to change
    // order between shapeCast and copy operation.
    // If shapeCast is moved before copy, instead of copying 4 channels,
    // copy operation will try to move 16 channels from memory.
    if (auto shapeCast = mlir::dyn_cast<VPUIP::ShapeCastOp>(*origOp)) {
        auto clusterTask = mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(*shapeCast.getResult().getUsers().begin());
        if (clusterTask != nullptr && clusterTask.getInputChannelsCompression() == true) {
            return mlir::failure();
        }
    }

    if (!VPUIP::getRootAlloc<mlir::memref::AllocOp>(copyOpOutput)) {
        _log.trace("Skip complex case: the operation defining the output buffer is not Alloc");
        return mlir::failure();
    }

    auto copyOpInputType = VPUIP::extractDataType(copyOpInput).cast<vpux::NDTypeInterface>();
    auto copyOpOutputType = VPUIP::extractDataType(copyOpOutput).cast<vpux::NDTypeInterface>();

    auto viewOpInputType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto viewOpOutputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto viewOpOutputShape = viewOpOutputType.getShape();
    auto viewOpOutputElemType = viewOpOutputType.getElementType();

    const auto inputShape = viewOpInputType.getShape();
    const auto outputShape = viewOpOutputType.getShape();
    const auto isRankChangedByViewOp = inputShape.size() != outputShape.size();
    auto distributedType = copyOpInput.getType().dyn_cast<VPUIP::DistributedBufferType>();
    mlir::FailureOr<std::pair<int64_t, int64_t>> getDistributedAxesMapping = mlir::failure();
    if (distributedType != nullptr && mlir::isa<VPUIP::ShapeCastOp, VPUIP::GenericReshapeOp>(origOp)) {
        getDistributedAxesMapping = getDistributedAxesMappingAfterShapeChanged(viewOpInputType, viewOpOutputType,
                                                                               distributedType.getDistribution(), _log);
    }

    const auto isSupportedDuplicated = [&](const VPU::DistributionMode& mode) {
        if (isRankChangedByViewOp && mlir::failed(getDistributedAxesMapping)) {
            return false;
        }

        return VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
               VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED);
    };
    if (distributedType != nullptr) {
        const auto isSupportSegmented = [&](const VPU::DistributionMode& mode) {
            // TODO: The num_tiles attribute also has to be adapted in case of different ranks
            if (isRankChangedByViewOp) {
                return false;
            }

            if (mode != VPU::DistributionMode::SEGMENTED || !VPU::isSegmentedOverH(distributedType.getDistribution())) {
                return false;
            }
            if (mlir::isa<VPUIP::QuantizeCastOp>(origOp)) {
                // Only support per-tensor uniform quantized type
                return (distributedType.getElementType().isa<mlir::quant::UniformQuantizedType>() &&
                        viewOpOutputElemType.isa<mlir::quant::UniformQuantizedType>());
            }
            // If the cluster copy op has siblings, moving pureViewOp
            // in front of it may cause accuracy issues
            if (!copyOpInput.hasOneUse()) {
                return false;
            }
            if (mlir::isa<VPUIP::PermuteCastOp>(origOp)) {
                // Currently only support SEGMENTED over H in NHWC layout,
                // so permuteCast to other order will break SOH and cannot
                // be applied on multicluster SEGMENTED.
                return false;
            }
            if (mlir::isa<VPUIP::ShapeCastOp, VPUIP::GenericReshapeOp>(origOp)) {
                const auto arch = VPU::getArch(origOp.getOperation());
                return VPUIP::isDistributedCompatibleAfterShapeChangeForViewOps(distributedType, viewOpOutputShape,
                                                                                viewOpOutputType.getDimsOrder(), arch);
            }
            return false;
        };
        const auto isSupportedOverlapping = [&](const VPUIP::DistributedBufferType distType,
                                                const mlir::ViewLikeOpInterface viewOp, const mlir::Value copyInput) {
            // TODO: The num_tiles attribute also has to be adapted in case of different ranks
            if (isRankChangedByViewOp) {
                return false;
            }

            auto distribution = distType.getDistribution();
            const auto mode = distribution.getMode().getValue();
            if (mode != VPU::DistributionMode::OVERLAPPED) {
                return false;
            }
            // If the cluster copy op has siblings, moving pureViewOp
            // in front of it may cause accuracy issues
            if (!copyInput.hasOneUse()) {
                return false;
            }
            if (mlir::isa<VPUIP::QuantizeCastOp>(viewOp)) {
                const auto viewOpOutputType = viewOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
                const auto viewOpOutputElemType = viewOpOutputType.getElementType();
                // Only support per-tensor uniform quantized type
                if (distType.getElementType().isa<mlir::quant::UniformQuantizedType>() &&
                    viewOpOutputElemType.isa<mlir::quant::UniformQuantizedType>()) {
                    return true;
                }
            }

            return false;
        };
        const auto mode = distributedType.getDistribution().getMode().getValue();
        if (!isSupportedDuplicated(mode) && !isSupportSegmented(mode) &&
            !isSupportedOverlapping(distributedType, origOp, copyOpInput)) {
            _log.trace("Not supported distributed type");
            return mlir::failure();
        }
    }

    // TODO: #62719
    const auto inReqs = StrideReqs::compact(copyOpInputType.getRank());
    if (!inReqs.checkStrides(copyOpInputType)) {
        _log.trace("Skip complex case: input is strided");
        return mlir::failure();
    }

    _log.trace("Set new input for '{0}': '{1}'", origOp->getName(), copyOpInput);
    origOp->setOperand(0, copyOpInput);

    vpux::NDTypeInterface newViewOpOutputType;

    auto getDistributionForViewOpOutput = [&]() -> VPU::DistributedTensorAttr {
        auto ctx = origOp->getContext();
        const auto arch = VPU::getArch(origOp.getOperation());
        const auto mode = distributedType.getDistribution().getMode().getValue();
        const auto origDistribution = distributedType.getDistribution();

        if (auto permuteCast = mlir::dyn_cast<VPUIP::PermuteCastOp>(*origOp)) {
            auto inPermuteType = permuteCast->getOperand(0).getType().cast<vpux::NDTypeInterface>();
            auto outPermuteType = permuteCast->getResult(0).getType().cast<vpux::NDTypeInterface>();

            return applyPermutationOnDistributedTensorAttr(origDistribution, permuteCast.getMemPerm(),
                                                           inPermuteType.getDimsOrder(), outPermuteType.getDimsOrder(),
                                                           inPermuteType.getShape(), outPermuteType.getShape());
        }

        if (mode == VPU::DistributionMode::SEGMENTED) {
            return VPUIP::getSOHDistAttrWithNewShape(ctx, distributedType, viewOpOutputShape, arch);
        }

        if (mlir::isa<VPUIP::ShapeCastOp, VPUIP::GenericReshapeOp>(origOp) && isSupportedDuplicated(mode)) {
            if (!VPU::isDistributedAttrWithExplicitShapesAndOffsets(origDistribution)) {
                if (isRankChangedByViewOp) {
                    auto axesMapping = getDistributedAxesMapping.value();
                    return changeDistributedAxisOnDistributedTensorAttr(origDistribution, axesMapping.first,
                                                                        axesMapping.second, viewOpOutputType.getRank());
                }
                return origDistribution;
            }

            // GenericReshape and ShapeCast can change the output shape without needing to follow any rule.
            // Therefore, when having distributions such as SEGMENTED|DUPLICATED or SEGMENTED|MULTICASTED
            // we might end up with the "tiling dim" not having the same shape it had at input. It is also possible for
            // the new shape to not be tile-able over the number of clusters.
            // However, GenericReshape & ShapeCast are ops that work on the memory view and do not need compute view
            // at all, so to ensure we do not end up with an output with a clustering dim that cannot be tiled, we're
            // setting distribution as DUPLICATED for output.
            auto duplicatedOutputMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
            return VPU::getNonOverlappedDistributedAttr(viewOpOutputShape, duplicatedOutputMode, nullptr,
                                                        origDistribution.getNumClusters(), nullptr,
                                                        origDistribution.getUniformDistributedSegments(), ctx);
        }

        return origDistribution;
    };

    if (distributedType != nullptr) {
        auto ctx = origOp->getContext();
        const auto order = mlir::AffineMapAttr::get(viewOpOutputType.getDimsOrder().toAffineMap(ctx));

        newViewOpOutputType =
                VPUIP::DistributedBufferType::get(ctx, viewOpOutputShape.raw(), viewOpOutputElemType, order,
                                                  distributedType.getMemSpace(), getDistributionForViewOpOutput());
    } else {
        newViewOpOutputType = viewOpOutputType.changeMemSpace(copyOpInputType.getMemSpace());
    }

    _log.trace("Set new result type for '{0}': '{1}'", origOp->getName(), newViewOpOutputType);
    origOp->getResult(0).setType(newViewOpOutputType);

    rewriter.setInsertionPointAfter(origOp);

    auto newAllocType = viewOpOutputType.changeMemSpace(copyOpOutputType.getMemSpace());
    auto allocOp = allocateBuffersOfType(_log, maybeCopy->getLoc(), rewriter, newAllocType).front();
    auto newCopyOp = _createNewCopyOp(rewriter, maybeCopy->getLoc(), origOp->getResult(0), allocOp);

    _log.trace("Replace all uses of pure view-like op with new Copy op: '{0}'", newCopyOp);
    origOp->getResult(0).replaceAllUsesExcept(newCopyOp->getResults()[0],
                                              llvm::SmallPtrSet<mlir::Operation*, 1>{newCopyOp});

    auto sourceOp = copyOpOutput.getDefiningOp();

    if (sourceOp != nullptr && sourceOp->getResult(0).use_empty()) {
        sourceOp->erase();
    }

    if (maybeCopy->getResult(0).use_empty()) {
        maybeCopy->erase();
    }

    return mlir::success();
}

//
// MoveSubviewToTheFrontOfCopy
//

class MoveViewOpToTheFrontOfCopy final : public LayerRewriterBase {
public:
    MoveViewOpToTheFrontOfCopy(mlir::MLIRContext* ctx, Logger log)
            : LayerRewriterBase(ctx, getCopyOp, createNewCopyOp, log) {
    }
};

//
// MoveViewOpToTheFrontOfTillingCopy
//

class MoveViewOpToTheFrontOfTillingCopy final : public LayerRewriterBase {
public:
    MoveViewOpToTheFrontOfTillingCopy(mlir::MLIRContext* ctx, Logger log)
            : LayerRewriterBase(ctx, getTillingCopyOp, createNewTillingCopyOp, log) {
    }
};

//
// MoveSubviewToTheFrontOfCopyBase
//
class MoveSubviewToTheFrontOfCopyBase : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    MoveSubviewToTheFrontOfCopyBase(mlir::MLIRContext* ctx, GetCopyFunctType getCopyOp,
                                    CreateCopyFunctType createNewCopyOp, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx),
              _getCopyOp(getCopyOp),
              _createNewCopyOp(createNewCopyOp),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    GetCopyFunctType _getCopyOp;
    CreateCopyFunctType _createNewCopyOp;
    Logger _log;
};

// SubView is not compatible with distributed buffer when:
// 1. Distributed buffer is segmented
// 2. SubView shrinks segmented axis
bool isSubViewCompatibleWithDistributedBuffer(VPUIP::SubViewOp subViewOp,
                                              VPUIP::DistributedBufferType distributedType) {
    const auto tileIndex = VPUIP::getTilingDimIndex(distributedType);
    if (!tileIndex.has_value()) {
        // DUPLICATED | MULTICASTED
        return true;
    }

    auto tileIndexVal = tileIndex.value();
    auto origShape = getShape(subViewOp.getSource());
    auto subShape = getShape(subViewOp.getResult());

    if (!VPUIP::isChannelOffsetsAndTileDimCompatibleWithClusterCopy(
                parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsetsAttr()), tileIndexVal, distributedType)) {
        return false;
    }

    // Be compatible if SubView does not shrink segmented axis
    return origShape[Dim(tileIndexVal)] == subShape[Dim(tileIndexVal)];
}

mlir::LogicalResult MoveSubviewToTheFrontOfCopyBase::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    auto subViewOp = copyOp.getInput().getDefiningOp<VPUIP::SubViewOp>();
    if (subViewOp == nullptr) {
        return mlir::failure();
    }

    auto sourceOp = subViewOp.getSource().getDefiningOp();
    if (sourceOp == nullptr) {
        // Source is BlockArgument
        return mlir::failure();
    }

    auto parentCopyOp = _getCopyOp(subViewOp.getSource().getDefiningOp());
    if (parentCopyOp == nullptr) {
        return mlir::failure();
    }

    // optimize happens only when tillingOp has one subview user
    if (!parentCopyOp->getResults()[0].hasOneUse()) {
        return mlir::failure();
    }

    // perform this optimization only when distributed buffer is compatible with subview
    // otherwise an accuracy degradation may occur
    auto originOperand = parentCopyOp->getOperand(0);
    if (auto distributedType = originOperand.getType().dyn_cast<VPUIP::DistributedBufferType>()) {
        if (!isSubViewCompatibleWithDistributedBuffer(subViewOp, distributedType)) {
            return mlir::failure();
        }
    }

    _log.trace("Move subview {0} in front of copy {1}", subViewOp->getLoc(), parentCopyOp->getLoc());

    if (auto arg = originOperand.dyn_cast<mlir::BlockArgument>()) {
        rewriter.setInsertionPointToStart(arg.getParentBlock());
    } else {
        rewriter.setInsertionPointAfter(originOperand.getDefiningOp());
    }

    // create and insert a new subview
    auto newSubViewOp =
            rewriter.create<VPUIP::SubViewOp>(subViewOp->getLoc(), originOperand, subViewOp.getStaticOffsetsAttr(),
                                              subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());

    auto subViewOpShape = getShape(newSubViewOp);
    auto allocOp = VPUIP::getRootAlloc<mlir::memref::AllocOp>(parentCopyOp.getOutputs()[0]);
    VPUX_THROW_UNLESS(allocOp, "CopyOp output buffer should be AllocationOp");
    auto allocOpDtype = allocOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    // Per-axis quantization must be aligned with the shape.
    const auto targetElemType = newSubViewOp.getResult().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto newAllocOpType = allocOpDtype.changeShapeElemType(subViewOpShape, targetElemType);
    if (mlir::isa<mlir::memref::AllocOp>(allocOp)) {
        allocOp->getResult(0).setType(allocOpDtype.changeShapeElemType(subViewOpShape, targetElemType));
    } else {
        mlir::OpBuilder::InsertPoint lastInsertionPoint = rewriter.saveInsertionPoint();
        rewriter.setInsertionPoint(allocOp);
        auto newAllocOp =
                allocateBuffersOfType(_log, allocOp->getLoc(), rewriter, newAllocOpType).front().getDefiningOp();
        rewriter.replaceOp(allocOp, newAllocOp->getResults());
        rewriter.restoreInsertionPoint(lastInsertionPoint);
        allocOp = newAllocOp;
    }

    auto newParentOp =
            _createNewCopyOp(rewriter, newSubViewOp->getLoc(), newSubViewOp.getResult(), allocOp->getResult(0));
    if (newParentOp->isBeforeInBlock(allocOp)) {
        VPUIP::moveRootAllocBefore(allocOp, newParentOp);
    }

    parentCopyOp->getResults()[0].replaceAllUsesWith(newParentOp->getResults()[0]);
    rewriter.eraseOp(parentCopyOp);

    // remove old subView
    subViewOp.getResult().replaceAllUsesWith(subViewOp.getSource());
    rewriter.eraseOp(subViewOp);
    return mlir::success();
}

//
// MoveSubviewToTheFrontOfCopy
//

/*
Move SubView to the front of Copy to make a chain of copies
     Copy(CMX2DDR)    =>          Subview
          |                          |
       Subview                  Copy(CMX2DDR)
          |                          |
        Copy                       Copy
*/

class MoveSubviewToTheFrontOfCopy final : public MoveSubviewToTheFrontOfCopyBase {
public:
    MoveSubviewToTheFrontOfCopy(mlir::MLIRContext* ctx, Logger log)
            : MoveSubviewToTheFrontOfCopyBase(ctx, getCopyOp, createNewCopyOp, log) {
    }
};

//
// MoveSubviewToTheFrontOfTillingCopy
//

/*
 Move SubView to the front of  TillingCopy, the assumption is copy src in CMX is faster than DDR
        NCEOp                      NCEOp
          |                          |
  TillingCopy(CMX2DDR)    =>      Subview
          |                          |
       Subview               TillingCopy(CMX2DDR)
          |                          |
        Copy                       Copy
*/

class MoveSubviewToTheFrontOfTillingCopy final : public MoveSubviewToTheFrontOfCopyBase {
public:
    MoveSubviewToTheFrontOfTillingCopy(mlir::MLIRContext* ctx, Logger log)
            : MoveSubviewToTheFrontOfCopyBase(ctx, getTillingCopyOp, createNewTillingCopyOp, log) {
    }
};

//
// MovePureViewOpBeforeCopyPass
//

class MovePureViewOpBeforeCopyPass final : public VPUIP::MovePureViewOpBeforeCopyBase<MovePureViewOpBeforeCopyPass> {
public:
    explicit MovePureViewOpBeforeCopyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MovePureViewOpBeforeCopyPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveViewOpToTheFrontOfCopy>(&ctx, _log);
    patterns.add<MoveViewOpToTheFrontOfTillingCopy>(&ctx, _log);
    patterns.add<MoveSubviewToTheFrontOfCopy>(&ctx, _log);
    patterns.add<MoveSubviewToTheFrontOfTillingCopy>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMovePureViewOpBeforeCopyPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createMovePureViewOpBeforeCopyPass(Logger log) {
    return std::make_unique<MovePureViewOpBeforeCopyPass>(log);
}

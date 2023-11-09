//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

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
        auto clusterTask = mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(*shapeCast.result().getUsers().begin());
        if (clusterTask != nullptr && clusterTask.input_channels_compression() == true) {
            return mlir::failure();
        }
    }

    if (!VPUIP::getRootAlloc<mlir::memref::AllocOp>(copyOpOutput)) {
        _log.trace("Skip complex case: the operation defining the output buffer is not Alloc");
        return mlir::failure();
    }

    auto copyOpInputType = VPUIP::extractDataType(copyOpInput).cast<vpux::NDTypeInterface>();
    auto copyOpOutputType = VPUIP::extractDataType(copyOpOutput).cast<vpux::NDTypeInterface>();

    auto viewOpOutputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto viewOpOutputShape = viewOpOutputType.getShape();
    auto viewOpOutputElemType = viewOpOutputType.getElementType();

    auto distributedType = copyOpInput.getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedType != nullptr) {
        const auto isDuplicated = [](const VPU::DistributionMode& mode) {
            return VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED) ||
                   VPU::bitEnumContains(mode, VPU::DistributionMode::MULTICASTED);
        };
        const auto isSupportSegmented = [&](const VPU::DistributionMode& mode) {
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
                const auto newDistributionAttr =
                        getSOHDistAttrWithNewShape(rewriter.getContext(), distributedType, viewOpOutputShape, arch);
                const auto tilingScheme = parseIntArrayAttr<int64_t>(newDistributionAttr.getNumTiles());
                const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
                const auto alignmentAttr = newDistributionAttr.getAlignment();
                if (alignmentAttr != nullptr) {
                    if (viewOpOutputShape[Dim(axis)] < parseIntArrayAttr<int64_t>(alignmentAttr)[axis]) {
                        return false;
                    }
                }
                return VPUIP::isDistributedCompatibleAfterShapeChange(distributedType, viewOpOutputShape, arch);
            }
            return false;
        };
        const auto isSupportedOverlapping = [](const VPUIP::DistributedBufferType distType,
                                               const mlir::ViewLikeOpInterface viewOp, const mlir::Value copyInput) {
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

            if (auto permuteCast = mlir::dyn_cast<VPUIP::PermuteCastOp>(*viewOp)) {
                auto inOrder = permuteCast->getOperand(0).getType().cast<vpux::NDTypeInterface>().getDimsOrder();
                auto outOrder = permuteCast->getResult(0).getType().cast<vpux::NDTypeInterface>().getDimsOrder();

                auto numTilesAttr = distribution.getNumTiles();
                const auto numTilesVec = parseIntArrayAttr<int64_t>(numTilesAttr);
                const auto numTilesInMemOrder = inOrder.toMemoryOrder(Shape(numTilesVec));
                const auto numTilesPermutedInMemOrder = applyPerm(numTilesInMemOrder, permuteCast.mem_perm());
                const auto permutedNumTiles = outOrder.toLogicalOrder(numTilesPermutedInMemOrder).raw();

                // Clustering axis will not be H or W after PermuteCast and will fail in OVERLAPPED distributed attr
                // check
                if (permutedNumTiles[Dims4D::Act::H.ind()] == 1 && permutedNumTiles[Dims4D::Act::W.ind()] == 1) {
                    return false;
                }
            }
            return false;
        };
        const auto mode = distributedType.getDistribution().getMode().getValue();
        if (!isDuplicated(mode) && !isSupportSegmented(mode) &&
            !isSupportedOverlapping(distributedType, origOp, copyOpInput)) {
            _log.trace("Not supported distributed type");
            return mlir::failure();
        }
        // TODO: The num_tiles attribute also has to be adapted in case of different ranks
        const auto inputShape = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getShape();
        const auto outputShape = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();
        if (inputShape.size() != outputShape.size()) {
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
            auto inOrder = permuteCast->getOperand(0).getType().cast<vpux::NDTypeInterface>().getDimsOrder();
            auto outOrder = permuteCast->getResult(0).getType().cast<vpux::NDTypeInterface>().getDimsOrder();

            return applyPermutationOnDistributedTensorAttr(origDistribution, permuteCast.mem_perm(), inOrder, outOrder);
        }

        return (mode == VPU::DistributionMode::SEGMENTED)
                       ? VPUIP::getSOHDistAttrWithNewShape(ctx, distributedType, viewOpOutputShape, arch)
                       : origDistribution;
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
    auto origShape = getShape(subViewOp.source());
    auto subShape = getShape(subViewOp.result());

    // Be compatible if SubView does not shrink segmented axis
    return origShape[Dim(tileIndexVal)] == subShape[Dim(tileIndexVal)];
}

mlir::LogicalResult MoveSubviewToTheFrontOfCopyBase::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    auto subViewOp = copyOp.input().getDefiningOp<VPUIP::SubViewOp>();
    if (subViewOp == nullptr) {
        return mlir::failure();
    }

    auto sourceOp = subViewOp.source().getDefiningOp();
    if (sourceOp == nullptr) {
        // Source is BlockArgument
        return mlir::failure();
    }

    auto parentCopyOp = _getCopyOp(subViewOp.source().getDefiningOp());
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
            rewriter.create<VPUIP::SubViewOp>(subViewOp->getLoc(), originOperand, subViewOp.static_offsetsAttr(),
                                              subViewOp.static_sizesAttr(), subViewOp.static_stridesAttr());

    auto subViewOpShape = getShape(newSubViewOp);
    auto allocOp = VPUIP::getRootAlloc<mlir::memref::AllocOp>(parentCopyOp.getOutputs()[0]);
    VPUX_THROW_UNLESS(allocOp, "CopyOp output buffer should be AllocationOp");
    auto allocOpDtype = allocOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    // Per-axis quantization must be aligned with the shape.
    const auto targetElemType = newSubViewOp.result().getType().cast<vpux::NDTypeInterface>().getElementType();
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

    auto newParentOp = _createNewCopyOp(rewriter, newSubViewOp->getLoc(), newSubViewOp.result(), allocOp->getResult(0));
    if (newParentOp->isBeforeInBlock(allocOp)) {
        VPUIP::moveRootAllocBefore(allocOp, newParentOp);
    }

    parentCopyOp->getResults()[0].replaceAllUsesWith(newParentOp->getResults()[0]);
    rewriter.eraseOp(parentCopyOp);

    // remove old subView
    subViewOp.result().replaceAllUsesWith(subViewOp.source());
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

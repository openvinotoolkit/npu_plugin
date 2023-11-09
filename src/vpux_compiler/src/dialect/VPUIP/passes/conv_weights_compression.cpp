//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>

#include <mlir/Pass/PassManager.h>

using namespace vpux;

namespace {

// We apply the reduce constant transformation only when we know for certain we have
// just padding over input channels. When having multiple Pads over different dimensions
// we found no clean way to be able to remove the padding for channels and recreate all
// the possible transformations that might come after it. So, for now, whenever we cannot
// remove the padding cleanly, we keep the constant as is and assume we're in a
// "input channels padded to 4" case. This is not ideal either, and we plan on
// implementing a separate compressed conv op in the future.
bool isOnlyPadOverC(Const::ContentAttr content) {
    const auto transformations = content.getTransformations();

    // Checks if the only padding applied is over IC dim
    for (auto& transform : transformations) {
        if (auto padWithZeroAttr = transform.dyn_cast<vpux::Const::PadWithZeroAttr>()) {
            const auto padAfter = parseIntArrayAttr<int64_t>(padWithZeroAttr.getPadAfter());
            const auto padBefore = parseIntArrayAttr<int64_t>(padWithZeroAttr.getPadBefore());

            // Weights alignment puts padding after, therefore we exclude all cases with padding
            // applied before.
            const bool hasNonZeroPadBefore = llvm::find_if(padBefore, [](int64_t pad) {
                                                 return pad != 0;
                                             }) != padBefore.end();
            if (hasNonZeroPadBefore || padAfter[Dims4D::Filter::KY.ind()] != 0 ||
                padAfter[Dims4D::Filter::KX.ind()] != 0) {
                return false;
            }
        }
    }

    return true;
}

mlir::Value reduceWeightsConstant(VPUIP::NCEClusterTaskOp nceOp, VPUIP::CopyOp weightsCopyOp,
                                  NDTypeInterface weightsCopyOutputType, ShapeRef origShape,
                                  const int64_t origChannelVal, mlir::OpBuilder& builder, bool isTiled) {
    const auto currentOffset = SmallVector<int64_t>{0, 0, 0, 0};
    mlir::Value weightsCopyInput =
            !isTiled ? weightsCopyOp.input()
                     : VPUIP::getTopBufferOfNCEClusterTiling(weightsCopyOp, weightsCopyOp.input());
    auto weightsConstOp = weightsCopyInput.getDefiningOp<vpux::Const::DeclareOp>();
    mlir::OpBuilder constBuilder(weightsConstOp);

    const auto newContentAttr = weightsConstOp.getContentAttr().subview(Shape(currentOffset), origShape);

    auto newConstType = weightsConstOp.getType().cast<NDTypeInterface>().changeShape(origShape);
    auto newWeightsConstOp = constBuilder.create<Const::DeclareOp>(
            weightsConstOp.getLoc(), newConstType.cast<mlir::MemRefType>(), newContentAttr);
    weightsConstOp.replaceAllUsesWith(newWeightsConstOp.getOperation());

    constexpr int64_t requiredAlignment = 16;
    const auto elemSize = getElemTypeSize(weightsCopyOutputType).count();

    // Check if weights set is aligned to 16B
    // If not, stride output of VPUIP.Copy accordingly
    if ((weightsCopyOutputType.getStrides()[Dims4D::Filter::OC].count() / elemSize) % requiredAlignment != 0) {
        auto paddedStrideValue = alignValUp(
                (origChannelVal * origShape[Dims4D::Filter::KY] * origShape[Dims4D::Filter::KX]), requiredAlignment);
        const SmallVector<Bit> newStrides({Bit(paddedStrideValue * elemSize), Bit(1 * elemSize),
                                           Bit(origChannelVal * origShape[Dims4D::Filter::KX] * elemSize),
                                           Bit(origChannelVal * elemSize)});
        weightsCopyOutputType = weightsCopyOutputType.changeStrides(vpux::StridesRef(newStrides));
    }

    if (isTiled) {
        auto weightsBufferTilingOp = weightsCopyOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
        auto oldDistrType = weightsBufferTilingOp.getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
        const auto orderAttr =
                mlir::AffineMapAttr::get(oldDistrType.getDimsOrder().toAffineMap(oldDistrType.getContext()));
        const auto elemStrides =
                to_small_vector(weightsCopyOutputType.getStrides() | transformed([&](Bit stride) {
                                    return stride.count() / weightsCopyOutputType.getElemTypeSize().count();
                                }));
        const auto stridesAttr = getIntArrayAttr(oldDistrType.getContext(), elemStrides);
        const auto layout = VPUIP::MemRefAttr::get(orderAttr, stridesAttr, nullptr, oldDistrType.getCompressionScheme(),
                                                   /*allocSize=*/nullptr, oldDistrType.getContext());
        auto distrib = VPUIP::DistributedBufferType::get(
                oldDistrType.getContext(), weightsCopyOutputType.getShape().raw(), oldDistrType.getElementType(),
                layout, oldDistrType.getMemSpace(), oldDistrType.getDistribution());

        auto copyAllocOp = VPUIP::getTopBufferOfNCEClusterTiling(weightsCopyOp, weightsCopyOp.output_buff())
                                   .getDefiningOp<VPURT::AllocDistributed>();
        mlir::IntegerAttr alignment = nullptr;
        if (copyAllocOp.getAlignmentAttr() != nullptr) {
            alignment = copyAllocOp.getAlignmentAttr();
        }
        mlir::Location loc = nceOp->getParentOfType<VPUIP::NCEClusterTilingOp>().getLoc();
        auto allocOp = builder.create<VPURT::AllocDistributed>(loc, distrib, alignment, nullptr);
        SmallVector<mlir::Value> inputsOutputCopyOperands = {newWeightsConstOp.getOutput(), allocOp};
        const auto copyOpBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
            builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
        };

        auto clusterTilingCopyOp =
                builder.create<VPUIP::NCEClusterTilingOp>(loc, distrib, inputsOutputCopyOperands, copyOpBodyBuilder);
        return clusterTilingCopyOp.results()[0];
    }

    auto allocOp =
            builder.create<mlir::memref::AllocOp>(nceOp->getLoc(), weightsCopyOutputType.cast<mlir::MemRefType>());

    auto copyOp = builder.create<VPUIP::CopyOp>(nceOp->getLoc(), newWeightsConstOp.getOutput(), allocOp);

    return copyOp.output();
}

mlir::Value getAdjustWeightsTable(Const::DeclareOp weightsTableConstOp, const int64_t weightsOC,
                                  const int64_t weightsOCstride) {
    auto weightsTableContent = weightsTableConstOp.getContent();
    auto weightsTableVals = to_std_vector(weightsTableContent.getValues<std::int32_t>());

    int64_t weightPtrOffset = 0;
    for (auto oc : irange(checked_cast<size_t>(weightsOC))) {
        const auto wtInd = oc * static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        weightsTableVals[wtInd] = checked_cast<int32_t>(weightPtrOffset);

        weightPtrOffset += weightsOCstride / CHAR_BIT;
    }

    mlir::OpBuilder constBuilder(weightsTableConstOp);
    const auto elemType = getSInt32Type(constBuilder.getContext());
    const auto weightTableShape = weightsTableContent.getType().getShape();
    const auto dataStorageType = mlir::RankedTensorType::get(weightTableShape.raw(), elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, makeArrayRef(weightsTableVals));

    auto dataConstOp = constBuilder.create<Const::DeclareOp>(
            weightsTableConstOp.getLoc(), weightsTableConstOp.getType(), Const::ContentAttr::get(dataAttr));

    return dataConstOp;
}

//
// CompressConvWeights
//

/*
    For weights with original IC < VPU_CHANNEL_ALIGNMENT, weights compression optimization can be applied.
    When we only have padding on weights ICs, we do the following:
        * strip weights of extra padding
        * set cm_sp_pattern for the original input channels
        * add ShapeCast to weights input of NCEClusterTask op
    Otherwise, we do:
        * set cm_sp_pattern for channels padded to VPU_COMPRESSED_INPUT_CHANNEL_NUM
        * add ShapeCast to weights input of NCEClusterTask op

    One case were we might have weights padded over another dimension is when weights IC <=
    VPU_COMPRESSED_INPUT_CHANNEL_NUM and we also have input activations compression. Then, the weights are padded up to
    VPU_COMPRESSED_INPUT_CHANNEL_NUM, not VPU_CHANNEL_ALIGNMENT. But that might lead to weights sets not being aligned
    to 16B. In that case, weights sets are flattened on W dimension and padded up to 16B. Therefore, the resulting shape
    will be: {OC x 1 x 1 x (IC padded up to VPU_COMPRESSED_INPUT_CHANNEL_NUM) * KY * KX + padding}

*/
void compressConvWeights(Logger& log, VPUIP::NCEClusterTaskOp origOp) {
    if (!VPUIP::canWeightsBeCompressed(origOp)) {
        return;
    }
    log.trace("Compressing weights for operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    mlir::OpBuilder builder(origOp);
    auto weights = origOp.weights().getDefiningOp<VPUIP::CopyOp>();
    auto weightsInput = weights.input().getDefiningOp<Const::DeclareOp>();
    auto weightsContentAttr = weightsInput.getContentAttr();

    const auto origChannelVal =
            weightsContentAttr.getBaseContent().getType().cast<NDTypeInterface>().getShape()[Dims4D::Filter::IC];
    const auto kernelSz = parseIntArrayAttr<int64_t>(origOp.kernel_sizeAttr());
    const auto outputChannels = origOp.output().getType().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];
    const auto origShape = Shape(
            {outputChannels, origChannelVal, kernelSz[Dims4D::Kernel::Y.ind()], kernelSz[Dims4D::Kernel::X.ind()]});

    auto weightsCopyOp = weights.output();
    auto weightsTableCopyOp = origOp.weight_table().getDefiningOp<VPUIP::CopyOp>();
    auto weightsTableConstOp = weightsTableCopyOp.input().getDefiningOp<Const::DeclareOp>();
    auto weightsMemRefType = weights.getType().cast<vpux::NDTypeInterface>();

    const bool paddingDoneOnlyOnC = isOnlyPadOverC(weightsContentAttr);
    if (paddingDoneOnlyOnC) {
        weightsMemRefType = weightsMemRefType.changeShape(origShape);

        weightsCopyOp =
                reduceWeightsConstant(origOp, weights, weightsMemRefType, origShape, origChannelVal, builder, false);

        // Removing the input channel padding for the weights const will lead to differences in
        // weight sets offsets. This will make the necessary adjustments.
        const auto weightsCopyOpDstStrides = weightsCopyOp.getType().cast<NDTypeInterface>().getStrides();
        auto newWeightsTableConst = getAdjustWeightsTable(weightsTableConstOp, origShape[Dims4D::Filter::OC],
                                                          weightsCopyOpDstStrides[Dims4D::Filter::OC].count());

        if (weightsTableConstOp.getOutput().hasOneUse()) {
            weightsTableConstOp.replaceAllUsesWith(newWeightsTableConst);
            weightsTableConstOp.erase();
        } else {
            // In this case, the weights table is shared between multiple ops,
            // compressing current op should not affect the others
            auto newWeightsTableCopy = builder.create<VPUIP::CopyOp>(weightsTableCopyOp->getLoc(), newWeightsTableConst,
                                                                     weightsTableCopyOp.output_buff());
            weightsTableCopyOp.replaceAllUsesWith(newWeightsTableCopy.getOperation());

            weightsTableCopyOp.erase();
        }
    }

    const auto channelAlignValue = VPU::NCEInvariant::getAlignment(weightsMemRefType.getElementType());
    const auto finalShape = SmallVector<int64_t>({origShape[Dims4D::Filter::OC], channelAlignValue,
                                                  origShape[Dims4D::Filter::KY], origShape[Dims4D::Filter::KX]});
    auto shapeCastOp = builder.create<VPUIP::ShapeCastOp>(origOp.getLoc(), weightsCopyOp,
                                                          getIntArrayAttr(origOp.getContext(), finalShape));

    const int64_t cmSpPattern =
            paddingDoneOnlyOnC ? (static_cast<int64_t>(1) << origChannelVal) - 1
                               : (static_cast<int64_t>(1) << VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM) - 1;
    auto cmSpPatternAttr = getIntAttr(origOp->getContext(), cmSpPattern);

    mlir::BlockAndValueMapping mapper;
    mapper.map(origOp.weights(), shapeCastOp.result());
    auto newOp = mlir::cast<VPUIP::NCEClusterTaskOp>(builder.clone(*origOp.getOperation(), mapper));
    newOp.cm_sp_patternAttr(cmSpPatternAttr);
    origOp.replaceAllUsesWith(newOp);
    origOp->erase();

    if (paddingDoneOnlyOnC) {
        weights->erase();
        weightsInput->erase();
    }
}

//
// compressClusterTiledConvWeights
//

void compressClusterTiledConvWeights(Logger& log, VPUIP::NCEClusterTilingOp origOp) {
    if (!VPUIP::canTilingWeightsBeCompressed(origOp)) {
        return;
    }

    log.trace("Compressing weights for operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    mlir::OpBuilder builder(origOp);
    auto nceOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(origOp.getInnerTaskOp());
    auto weights = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.weights());
    auto weightsBufferTilingOp = weights.getDefiningOp<VPUIP::NCEClusterTilingOp>();
    auto weightsCopyOp = weightsBufferTilingOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
    auto weightsInput = VPUIP::getTopBufferOfNCEClusterTiling(weightsCopyOp, weightsCopyOp.input())
                                .getDefiningOp<Const::DeclareOp>();
    auto weightsContentAttr = weightsInput.getContentAttr();

    const auto origChannelVal =
            weightsContentAttr.getBaseContent().getType().cast<NDTypeInterface>().getShape()[Dims4D::Filter::IC];
    const auto kernelSz = parseIntArrayAttr<int64_t>(nceOp.kernel_sizeAttr());
    const auto outputChannels = nceOp.output().getType().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];
    const auto origShape = Shape(
            {outputChannels, origChannelVal, kernelSz[Dims4D::Kernel::Y.ind()], kernelSz[Dims4D::Kernel::X.ind()]});

    auto weightsTable = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.weight_table());
    auto weightsTableBufferTilingOp = weightsTable.getDefiningOp<VPUIP::NCEClusterTilingOp>();
    auto weightsTableCopyOp = weightsTableBufferTilingOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
    auto weightsTableConstOp = VPUIP::getTopBufferOfNCEClusterTiling(weightsTableCopyOp, weightsTableCopyOp.input())
                                       .getDefiningOp<Const::DeclareOp>();

    mlir::Value weightsOutCopyOp = weightsBufferTilingOp->getResult(0);
    auto weightsMemRefType = weightsCopyOp.getType().cast<vpux::NDTypeInterface>();

    const bool paddingDoneOnlyOnC = isOnlyPadOverC(weightsContentAttr);
    if (paddingDoneOnlyOnC) {
        weightsMemRefType = weightsMemRefType.changeShape(origShape);

        weightsOutCopyOp = reduceWeightsConstant(nceOp, weightsCopyOp, weightsMemRefType, origShape, origChannelVal,
                                                 builder, true);

        // Removing the input channel padding for the weights const will lead to differences in
        // weight sets offsets. This will make the necessary adjustments.
        const auto weightsCopyOpDstStrides = weightsOutCopyOp.getType().cast<NDTypeInterface>().getStrides();
        auto newWeightsTableConst = getAdjustWeightsTable(weightsTableConstOp, origShape[Dims4D::Filter::OC],
                                                          weightsCopyOpDstStrides[Dims4D::Filter::OC].count());

        if (weightsTableConstOp.getOutput().hasOneUse()) {
            weightsTableConstOp.replaceAllUsesWith(newWeightsTableConst);
            weightsTableConstOp.erase();
        } else {
            // In this case, the weights table is shared between multiple ops,
            // compressing current op should not affect the others
            SmallVector<mlir::Value> inputsOutputOperands = {newWeightsTableConst,
                                                             weightsTableBufferTilingOp.output_buffs()[0]};
            const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
                builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
            };
            auto newWeightsTableBufferTiling = builder.create<VPUIP::NCEClusterTilingOp>(
                    weightsTableBufferTilingOp.getLoc(), weightsTableBufferTilingOp.getResultTypes(),
                    inputsOutputOperands, bodyBuilder);
            weightsTableBufferTilingOp.replaceAllUsesWith(newWeightsTableBufferTiling);

            weightsTableCopyOp.erase();
            weightsTableBufferTilingOp.erase();
        }
    }

    const auto channelAlignValue = VPU::NCEInvariant::getAlignment(weightsMemRefType.getElementType());
    const auto finalShape = SmallVector<int64_t>({origShape[Dims4D::Filter::OC], channelAlignValue,
                                                  origShape[Dims4D::Filter::KY], origShape[Dims4D::Filter::KX]});
    auto shapeCastOp = builder.create<VPUIP::ShapeCastOp>(origOp.getLoc(), weightsOutCopyOp,
                                                          getIntArrayAttr(origOp.getContext(), finalShape));

    const int64_t cmSpPattern =
            paddingDoneOnlyOnC ? (static_cast<int64_t>(1) << origChannelVal) - 1
                               : (static_cast<int64_t>(1) << VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM) - 1;
    auto cmSpPatternAttr = getIntAttr(origOp->getContext(), cmSpPattern);

    auto weightsBlockArg = nceOp.weights().dyn_cast<mlir::BlockArgument>();
    SmallVector<mlir::Value> operands = origOp.getOperands();
    operands[weightsBlockArg.getArgNumber()] = shapeCastOp.result();

    const auto nceOpBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location /*loc*/, mlir::ValueRange newOperands) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(origOp.body().front().getArguments(), newOperands);
        auto newOp = builder.clone(*nceOp.getOperation(), mapper);
        mlir::cast<VPUIP::NCEClusterTaskOp>(newOp).cm_sp_patternAttr(cmSpPatternAttr);
    };
    auto clusterTilingNceOp = builder.create<VPUIP::NCEClusterTilingOp>(origOp->getLoc(), origOp->getResultTypes(),
                                                                        mlir::ValueRange{operands}, nceOpBodyBuilder);

    origOp.replaceAllUsesWith(clusterTilingNceOp);
    origOp.getInnerTaskOp()->erase();
    origOp->erase();
    if (paddingDoneOnlyOnC) {
        weightsCopyOp->erase();
        weightsBufferTilingOp->erase();
        weightsInput->erase();
    }
}

//
// ConvWeightsCompression
//

class ConvWeightsCompression final : public VPUIP::ConvWeightsCompressionBase<ConvWeightsCompression> {
public:
    explicit ConvWeightsCompression(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvWeightsCompression::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUIP::NCEClusterTaskOp origOp) {
        compressConvWeights(_log, origOp);
    });
    func.walk([&](VPUIP::NCEClusterTilingOp origOp) {
        compressClusterTiledConvWeights(_log, origOp);
    });
}

}  // namespace

//
// createConvWeightsCompressionPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvWeightsCompressionPass(Logger log) {
    return std::make_unique<ConvWeightsCompression>(log);
}

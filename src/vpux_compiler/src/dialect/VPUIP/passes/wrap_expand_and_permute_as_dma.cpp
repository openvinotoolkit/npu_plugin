
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>

using namespace vpux;
namespace {

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef shape, ShapeRef offset) {
    const auto elemType = originType.getElementType();
    if (auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto newQType = tileScalesAndZP(qType, shape, offset);
        return originType.changeShapeElemType(shape, newQType);
    }

    return originType.changeShape(shape);
}

bool isSplitContinuousBufferType(vpux::NDTypeInterface innerType, VPUIP::DistributedBufferType distributedType) {
    auto isCompactType = [](vpux::NDTypeInterface origType) {
        const auto shape = origType.getShape();
        const auto strideReqs = StrideReqs::compact(shape.size());
        return strideReqs.checkStrides(origType);
    };
    if (!isCompactType(distributedType)) {
        return false;
    }

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.num_clusters().getInt();
    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    const auto tileInnerType = [&](vpux::NDTypeInterface innerType) {
        SmallVector<vpux::NDTypeInterface> newTypes(numClusters);
        for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
            newTypes[clusterId] =
                    changeShape(innerType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
        }

        return newTypes;
    };
    auto outTypes = tileInnerType(innerType);
    return llvm::all_of(outTypes, isCompactType);
}

// Check pattern 1: Copy(ddr->cmx) -> sw.kernel(memPermute) -> Copy (cmx-> ddr) -> Copy (ddr-> cmx)
// Check pattern 2: Copy(ddr->cmx) -> sw.kernel(memPermute) -> Copy (cmx-> ddr) -> ClusterCopy (ddr-> cmx)
bool checkPermutePattern(VPUIP::SwKernelOp swKernelOp, Logger log) {
    if (!VPUIP::isMemPermSwKernel(swKernelOp)) {
        return false;
    }

    log.trace("Got MemPermute SwKernel at {0}. Try to find fuse pattern.", swKernelOp->getLoc());

    if (!VPUIP::isLegalConvertToDMA(swKernelOp, log)) {
        log.nest().trace("VPUIP.SwKernel can not be converted to DMA at {0}", swKernelOp->getLoc());
        return false;
    }

    auto permuteInType = swKernelOp.getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
    auto permuteOutType = swKernelOp.getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();

    if (!swKernelOp->hasOneUse()) {
        log.nest().trace("VPUIP.SwKernel has more than one use at {0}", swKernelOp->getLoc());
        return false;
    }

    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    if (copyInCMXOp == nullptr || !copyInCMXOp->hasOneUse()) {
        return false;
    }

    const auto copyInInputType = copyInCMXOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto copyInOutputType = copyInCMXOp.output().getType().cast<vpux::NDTypeInterface>();
    if (copyInInputType.getMemoryKind() != VPU::MemoryKind::DDR ||
        copyInOutputType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
        return false;
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (copyBackToDDROp == nullptr || !copyBackToDDROp->hasOneUse()) {
        return false;
    }

    const auto copyBackInputType = copyBackToDDROp.input().getType().cast<vpux::NDTypeInterface>();
    const auto copyBackOutputType = copyBackToDDROp.output().getType().cast<vpux::NDTypeInterface>();
    if (copyBackInputType.getMemoryKind() != VPU::MemoryKind::CMX_NN ||
        copyBackOutputType.getMemoryKind() != VPU::MemoryKind::DDR) {
        return false;
    }

    auto copyToNCEOp = *(copyBackToDDROp->getUsers().begin());
    if (auto clusterOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(copyToNCEOp)) {
        // It is difficult to use the general method to fuse Permute the with next Cluster Copy Op
        // which has the stride. For example, activation with NHWC layout, need tile at Channel.
        // It is necessary to check the split buffer is continuous.
        const auto clusterOutput = *clusterOp.getOutputs().begin();
        const auto clusterOutputType = clusterOutput.getType().cast<vpux::NDTypeInterface>();

        const auto innerOutput = *clusterOp.getInnerOutputs().begin();
        const auto innerOutputType = innerOutput.getType();
        if (!isSplitContinuousBufferType(innerOutputType, clusterOutputType.dyn_cast<VPUIP::DistributedBufferType>())) {
            return false;
        }

        auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp).getValue();
        auto dmaSubShapes = VPUIP::getPermuteDMASubInputShapes(permuteInType, permuteOutType, memPerm, log);
        // If fuse Permute with next Cluster Copy Op and PermuteDMA need unroll to severl Sub DMA tasks,
        // Find a scenerior has regression. Need investigate the root cause and find a cost model for that.
        // For example: Shape size with 1x4420x1x2, mode is DUPLICATED.
        // It will be unrolled to 18 PermuteDMA with shape size 1x256x1x2 (17) + 1x68x1x2 (1)
        if (!dmaSubShapes.hasValue() || dmaSubShapes.getValue().size() > 1) {
            return false;
        }

        if (!VPUIP::doesPermuteDMATileDimSupportWrapInCluster(
                    permuteInType, permuteOutType, memPerm, clusterOutputType.dyn_cast<VPUIP::DistributedBufferType>(),
                    log)) {
            return false;
        }

        copyToNCEOp = clusterOp.getInnerTaskOp();
    }

    auto childCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(copyToNCEOp);
    if (childCopyOp == nullptr) {
        return false;
    }

    const auto childOutputType = childCopyOp.output().getType().cast<vpux::NDTypeInterface>();
    return childOutputType.getMemoryKind() == VPU::MemoryKind::CMX_NN;
}

// Check pattern 1: Expand(ddr->ddr) -> Copy (ddr->cmx) (U8 precision)
// Check pattern 2: Expand(ddr->ddr) -> ClusterCopy (ddr->cmx) (U8 precision)
bool checkExpandPattern(VPUIP::ExpandOp expandOp, Logger log) {
    log.trace("Got ExpandOp at {0}. Try to find fuse pattern.", expandOp->getLoc());

    // When exapnd with U8 precision, unitialized data can be used
    // When exapnd with FP16 precision, still need to duplicate the data
    const auto elemType = expandOp.output().getType().cast<vpux::NDTypeInterface>();
    if (!elemType.getElementType().isa<mlir::quant::QuantizedType>()) {
        log.nest().trace("ExpandOp convert to DMA shloud with U8 precision.");
        return false;
    }

    if (elemType.getDimsOrder() != DimsOrder::NCHW && elemType.getDimsOrder() != DimsOrder::NHWC) {
        log.nest().trace("ExpandOp convert to DMA shloud with NCHW or NHWC layout.");
        return false;
    }

    const auto nonZeroAxisPredicate = [](const int64_t dim) -> bool {
        return dim > 0;
    };

    const auto hasPadAndPadAtChannel = [&](mlir::ArrayAttr pads) -> bool {
        const auto padsValue = parseIntArrayAttr<int64_t>(pads);
        const auto padAxisIter = std::find_if(padsValue.begin(), padsValue.end(), nonZeroAxisPredicate);
        if (padAxisIter != padsValue.end()) {
            const auto padAxis = std::distance(padsValue.begin(), padAxisIter);
            return padAxis == Dims4D::Act::C.ind();
        }
        return false;
    };

    // Only support Expand layer with padding at channel and padding at end
    // TODO: Padding at any axis
    const auto padBegin = parseIntArrayAttr<int64_t>(expandOp.pads_begin());
    if (std::any_of(padBegin.begin(), padBegin.end(), nonZeroAxisPredicate)) {
        log.nest().trace("Only support Expand layer with padding at the end. But got {0}.", padBegin);
        return false;
    }

    if (!hasPadAndPadAtChannel(expandOp.pads_end())) {
        log.nest().trace("Only support Expand layer with padding at channel. But got {0}.", expandOp.pads_end());
        return false;
    }

    if (!expandOp->hasOneUse()) {
        return false;
    }

    auto copyOutOp = *(expandOp->getUsers().begin());
    if (auto clusterOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(copyOutOp)) {
        // It is difficult to use the general method to fuse Expand the with next Cluster Copy Op
        // which has the stride. For example, activation with NHWC layout, need tile at Channel.
        // It is necessary to check the split buffer is continuous.
        const auto clusterOutput = *clusterOp.getOutputs().begin();
        const auto clusterOutputType = clusterOutput.getType().cast<vpux::NDTypeInterface>();
        const auto innerOutput = *clusterOp.getInnerOutputs().begin();
        const auto innerOutputType = innerOutput.getType();
        if (!isSplitContinuousBufferType(innerOutputType, clusterOutputType.dyn_cast<VPUIP::DistributedBufferType>())) {
            return false;
        }

        copyOutOp = clusterOp.getInnerTaskOp();
    }

    auto childCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(copyOutOp);
    if (childCopyOp == nullptr) {
        return false;
    }

    const auto copyOutInputType = childCopyOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto copyOutOutputType = childCopyOp.output().getType().cast<vpux::NDTypeInterface>();
    if (copyOutInputType.getMemoryKind() != VPU::MemoryKind::DDR ||
        copyOutOutputType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
        return false;
    }

    return true;
}

// Check pattern 1: Expand (NCHW) -> Copy(ddr->cmx) -> sw.kernel(memPermute to NHWC) -> Copy (cmx-> ddr)
//                   -> Copy (ddr-> cmx) (U8 precision)
// Check pattern 2: Expand (NCHW) -> Copy(ddr->cmx) -> sw.kernel(memPermute to NHWC) -> Copy (cmx-> ddr)
//                   -> ClusterCopy (ddr-> cmx) (U8 precision)
bool checkExpandWithPermutePattern(VPUIP::ExpandOp expandOp, Logger log) {
    log.trace("Got ExpandOp at {0}. Try to find fuse expand and permute pattern.", expandOp->getLoc());

    // Just support Expand with layout NCHW
    const auto inOrder = DimsOrder::fromValue(expandOp.input());
    if (inOrder != DimsOrder::NCHW) {
        log.nest().trace("Expand With Permute Pattern should with NCHW layout. Got {0}.", inOrder);
        return false;
    }

    if (!checkExpandPattern(expandOp, log)) {
        return false;
    }

    auto expandUserOp = *(expandOp->getUsers().begin());
    if (mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(expandUserOp)) {
        return false;
    }

    auto expandCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(expandUserOp);
    if (expandCopyOp == nullptr || !expandOp->hasOneUse()) {
        return false;
    }

    auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(*(expandCopyOp->getUsers().begin()));
    if (swKernelOp == nullptr || !swKernelOp->hasOneUse()) {
        return false;
    }

    // Just support Permute with input layout NCHW and output layout NHWC
    const auto permuteInOrder = DimsOrder::fromValue(swKernelOp.inputs().front());
    const auto permuteOutOrder = DimsOrder::fromValue(swKernelOp.getOutputs().front());
    if (permuteInOrder != DimsOrder::NCHW || permuteOutOrder != DimsOrder::NHWC) {
        log.nest().trace("Just support Permute with input layout NCHW and output layout NHWC. Got {0}, {1}.",
                         permuteInOrder, permuteOutOrder);
        return false;
    }

    return checkPermutePattern(swKernelOp, log);
}

//
// FuseMemPermuteWithClusterCopy
//

// Copy(ddr->cmx)
//      |
// SW.kernel(memPermute)
//      |                                 ->        Clustering Tiling PermuteDMA (ddr->cmx)
// Copy (cmx->ddr)
//      |
// Clustering Ciling Copy (ddr->cmx)

class FuseMemPermuteWithClusterCopy final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FuseMemPermuteWithClusterCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FuseMemPermuteWithClusterCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swkernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseMemPermuteWithClusterCopy::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    if (!checkPermutePattern(swKernelOp, _log)) {
        return mlir::failure();
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (copyBackToDDROp == nullptr) {
        return mlir::failure();
    }

    auto clusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*(copyBackToDDROp->getUsers().begin()));
    if (clusterCopyOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got Permute -> Cluster Copy pattern. MemPermute '{0}' at '{1}'", swKernelOp->getName(),
               swKernelOp->getLoc());

    // Check distribution mode
    const auto clusterOutput = *clusterCopyOp.getOutputs().begin();
    const auto clusterOutputType = clusterOutput.getType().cast<vpux::NDTypeInterface>();
    const auto distributedType = clusterOutputType.dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedType == nullptr) {
        return mlir::failure();
    }

    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.mode().getValue();
    if (mode != VPU::DistributionMode::SEGMENTED && mode != VPU::DistributionMode::OVERLAPPED &&
        !VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED)) {
        return mlir::failure();
    }

    auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp).getValue();
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::PermuteDMAOp>(swKernelOp->getLoc(), newOperands[0], newOperands[1],
                                            mlir::AffineMapAttr::get(memPerm), nullptr);
    };

    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();

    SmallVector<mlir::Value> newNceClusterTilingOperands;
    newNceClusterTilingOperands.push_back(copyInCMXOp.input());
    newNceClusterTilingOperands.push_back(clusterCopyOp.output_buffs()[0]);

    rewriter.setInsertionPointAfter(clusterCopyOp);
    rewriter.replaceOpWithNewOp<VPUIP::NCEClusterTilingOp>(clusterCopyOp, clusterOutputType,
                                                           newNceClusterTilingOperands, bodyBuilder);

    _log.nest().trace("Wrap MemPermute '{0}' at '{1}' with next Cluster Copy.", swKernelOp->getName(),
                      swKernelOp->getLoc());

    rewriter.eraseOp(copyBackToDDROp);
    rewriter.eraseOp(swKernelOp);
    rewriter.eraseOp(copyInCMXOp);

    return mlir::success();
}

//
// FuseMemPermuteWithCopy
//

// Copy(ddr->cmx)
//      |
// SW.kernel(memPermute)
//      |                      ->     VPUIP.PermuteDMA(ddr->cmx)
// Copy (cmx->ddr)
//      |
// Copy (ddr->cmx)

class FuseMemPermuteWithCopy final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FuseMemPermuteWithCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FuseMemPermuteWithCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swkernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseMemPermuteWithCopy::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                            mlir::PatternRewriter& rewriter) const {
    if (!checkPermutePattern(swKernelOp, _log)) {
        return mlir::failure();
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    auto childCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*(copyBackToDDROp->getUsers().begin()));
    if (childCopyOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got Permute -> Copy pattern. MemPermute '{0}' at '{1}'", swKernelOp->getName(), swKernelOp->getLoc());

    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp).getValue();

    rewriter.setInsertionPointAfter(childCopyOp);
    rewriter.replaceOpWithNewOp<VPUIP::PermuteDMAOp>(childCopyOp, copyInCMXOp.input(), childCopyOp.output_buff(),
                                                     mlir::AffineMapAttr::get(memPerm), nullptr);

    _log.nest().trace("Wrap MemPermute '{0}' at '{1}' with next copy.", swKernelOp->getName(), swKernelOp->getLoc());

    rewriter.eraseOp(copyBackToDDROp);
    rewriter.eraseOp(swKernelOp);
    rewriter.eraseOp(copyInCMXOp);

    return mlir::success();
}

//
// FuseExpandWithClusterCopy
//

// Expand (U8)
//      |                                 ->      Cluster Tiling ExpandDMA (ddr->cmx)
// Cluster Tiling Copy (ddr->cmx)

class FuseExpandWithClusterCopy final : public mlir::OpRewritePattern<VPUIP::ExpandOp> {
public:
    FuseExpandWithClusterCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ExpandOp>(ctx), _log(log) {
        setDebugName("FuseExpandWithClusterCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseExpandWithClusterCopy::matchAndRewrite(VPUIP::ExpandOp expandOp,
                                                               mlir::PatternRewriter& rewriter) const {
    if (!checkExpandPattern(expandOp, _log)) {
        return mlir::failure();
    }

    auto clusterCopyOutOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*(expandOp->getUsers().begin()));
    if (clusterCopyOutOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got Expand -> Cluster Copy pattern. Expand '{0}' at '{1}'", expandOp->getName(), expandOp->getLoc());

    // check distribution mode
    const auto clusterOutput = *clusterCopyOutOp.getOutputs().begin();
    const auto clusterOutputType = clusterOutput.getType().cast<vpux::NDTypeInterface>();
    const auto distributedType = clusterOutputType.dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedType == nullptr) {
        return mlir::failure();
    }

    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.mode().getValue();
    if (mode != VPU::DistributionMode::SEGMENTED && mode != VPU::DistributionMode::OVERLAPPED &&
        !VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED)) {
        return mlir::failure();
    }

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::ExpandDMAOp>(expandOp->getLoc(), newOperands[0], newOperands[1],
                                           expandOp.pads_beginAttr(), expandOp.pads_endAttr());
    };

    SmallVector<mlir::Value> newNceClusterTilingOperands;
    newNceClusterTilingOperands.push_back(expandOp.input());
    newNceClusterTilingOperands.push_back(clusterCopyOutOp.output_buffs()[0]);

    rewriter.setInsertionPointAfter(clusterCopyOutOp);
    rewriter.replaceOpWithNewOp<VPUIP::NCEClusterTilingOp>(clusterCopyOutOp, clusterOutputType,
                                                           newNceClusterTilingOperands, bodyBuilder);

    _log.nest().trace("Wrap Expand '{0}' at '{1}' with next Cluster Copy.", expandOp->getName(), expandOp->getLoc());

    rewriter.eraseOp(expandOp);

    return mlir::success();
}

//
// FuseExpandWithCopy
//

// Expand (U8)
//      |                ->      ExpandDMA (ddr->cmx)
// Copy (ddr->cmx)

class FuseExpandWithCopy final : public mlir::OpRewritePattern<VPUIP::ExpandOp> {
public:
    FuseExpandWithCopy(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::ExpandOp>(ctx), _log(log) {
        setDebugName("FuseExpandWithCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseExpandWithCopy::matchAndRewrite(VPUIP::ExpandOp expandOp,
                                                        mlir::PatternRewriter& rewriter) const {
    if (!checkExpandPattern(expandOp, _log)) {
        return mlir::failure();
    }

    _log.trace("Got Expand -> Copy pattern. Expand '{0}' at '{1}'", expandOp->getName(), expandOp->getLoc());

    auto expandCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(expandOp->getUsers().begin()));
    if (expandCopyOutOp == nullptr) {
        return mlir::failure();
    }

    rewriter.setInsertionPointAfter(expandCopyOutOp);
    rewriter.replaceOpWithNewOp<VPUIP::ExpandDMAOp>(expandCopyOutOp, expandOp.input(), expandCopyOutOp.output_buff(),
                                                    expandOp.pads_beginAttr(), expandOp.pads_endAttr());

    _log.nest().trace("Wrap Expand '{0}' at '{1}' with next Copy.", expandOp->getName(), expandOp->getLoc());

    rewriter.eraseOp(expandOp);

    return mlir::success();
}

//
// FuseExpandAndPermuteWithClusterCopy
//

// Expand (U8)
//      |
// Copy(ddr->cmx)
//      |
// SW.kernel(memPermute)          ->       Clustering Tiling PermuteDMA (ddr->cmx)
//      |
// Copy (cmx->ddr)
//      |
// Clustering Tiling Copy (ddr->cmx)

class FuseExpandAndPermuteWithClusterCopy final : public mlir::OpRewritePattern<VPUIP::ExpandOp> {
public:
    FuseExpandAndPermuteWithClusterCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ExpandOp>(ctx), _log(log) {
        setDebugName("FuseExpandAndPermuteWithClusterCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseExpandAndPermuteWithClusterCopy::matchAndRewrite(VPUIP::ExpandOp expandOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    if (!checkExpandWithPermutePattern(expandOp, _log)) {
        return mlir::failure();
    }

    auto expandCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(expandOp->getUsers().begin()));
    auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(*(expandCopyOutOp->getUsers().begin()));

    auto permuteCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    auto clusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*(permuteCopyOutOp->getUsers().begin()));
    if (clusterCopyOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got Expand -> permute -> Cluster Copy pattern. Expand '{0}' at '{1}'", expandOp->getName(),
               expandOp->getLoc());

    // check distribution mode
    const auto clusterCopyOutput = *clusterCopyOp.getOutputs().begin();
    const auto clusterOutputType = clusterCopyOutput.getType().cast<vpux::NDTypeInterface>();
    const auto distributedType = clusterOutputType.dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedType == nullptr) {
        return mlir::failure();
    }

    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.mode().getValue();
    if (mode != VPU::DistributionMode::SEGMENTED && mode != VPU::DistributionMode::OVERLAPPED &&
        !VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED)) {
        return mlir::failure();
    }

    auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp).getValue();
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::PermuteDMAOp>(swKernelOp->getLoc(), newOperands[0], newOperands[1],
                                            mlir::AffineMapAttr::get(memPerm), nullptr);
    };

    SmallVector<mlir::Value> newNceClusterTilingOperands;
    newNceClusterTilingOperands.push_back(expandOp.input());
    newNceClusterTilingOperands.push_back(clusterCopyOp.output_buffs()[0]);

    rewriter.setInsertionPointAfter(clusterCopyOp);
    rewriter.replaceOpWithNewOp<VPUIP::NCEClusterTilingOp>(clusterCopyOp, clusterOutputType,
                                                           newNceClusterTilingOperands, bodyBuilder);

    _log.nest().trace("Wrap Expand '{0}' at '{1}' and MemPermute with next Cluster Copy.", expandOp->getName(),
                      expandOp->getLoc());

    rewriter.eraseOp(permuteCopyOutOp);
    rewriter.eraseOp(swKernelOp);
    rewriter.eraseOp(expandCopyOutOp);
    rewriter.eraseOp(expandOp);

    return mlir::success();
}

//
// FuseExpandAndPermuteWithCopy
//

// Expand (U8)
//      |
// Copy(ddr->cmx)
//      |
// SW.kernel(memPermute)         ->      PermuteDMA (ddr->cmx)
//      |
// Copy (cmx->ddr)
//      |
// Copy (ddr->cmx)

class FuseExpandAndPermuteWithCopy final : public mlir::OpRewritePattern<VPUIP::ExpandOp> {
public:
    FuseExpandAndPermuteWithCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ExpandOp>(ctx), _log(log) {
        setDebugName("FuseExpandAndPermuteWithCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandOp expandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseExpandAndPermuteWithCopy::matchAndRewrite(VPUIP::ExpandOp expandOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    if (!checkExpandWithPermutePattern(expandOp, _log)) {
        return mlir::failure();
    }

    auto expandCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(expandOp->getUsers().begin()));
    auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(*(expandCopyOutOp->getUsers().begin()));

    auto permuteCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    auto childCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*(permuteCopyOutOp->getUsers().begin()));
    if (childCopyOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got Expand -> Permute -> Copy pattern. Expand '{0}' at '{1}'", expandOp->getName(), expandOp->getLoc());

    rewriter.setInsertionPointAfter(childCopyOp);
    auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp).getValue();
    rewriter.replaceOpWithNewOp<VPUIP::PermuteDMAOp>(childCopyOp, expandOp.input(), childCopyOp.output_buff(),
                                                     mlir::AffineMapAttr::get(memPerm), nullptr);

    _log.nest().trace("Wrap Expand '{0}' at '{1}' and MemPermute with next Copy.", expandOp->getName(),
                      expandOp->getLoc());

    rewriter.eraseOp(permuteCopyOutOp);
    rewriter.eraseOp(swKernelOp);
    rewriter.eraseOp(expandCopyOutOp);
    rewriter.eraseOp(expandOp);

    return mlir::success();
}

//
// WrapExpandAndPermuteAsDMAPass
//

class WrapExpandAndPermuteAsDMAPass final : public VPUIP::WrapExpandAndPermuteAsDMABase<WrapExpandAndPermuteAsDMAPass> {
public:
    explicit WrapExpandAndPermuteAsDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void WrapExpandAndPermuteAsDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto module = getOperation();
    const auto arch = VPU::getArch(module);
    if (arch != VPU::ArchKind::VPUX37XX) {
        _log.trace("WrapExpandAndPermuteAsDMAPass enabled only for VPUX37XX device. Got: {0}", arch);
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseExpandAndPermuteWithClusterCopy>(&ctx, _log);
    patterns.add<FuseExpandAndPermuteWithCopy>(&ctx, _log);
    patterns.add<FuseExpandWithClusterCopy>(&ctx, _log);
    patterns.add<FuseExpandWithCopy>(&ctx, _log);
    patterns.add<FuseMemPermuteWithClusterCopy>(&ctx, _log);
    patterns.add<FuseMemPermuteWithCopy>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createWrapExpandAndPermuteAsDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createWrapExpandAndPermuteAsDMAPass(Logger log) {
    return std::make_unique<WrapExpandAndPermuteAsDMAPass>(log);
}


// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
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

    if (!isCompactType(distributedType) || !isMemoryContiguousWithTiling(distributedType)) {
        return false;
    }

    const auto distributionAttr = distributedType.getDistribution();
    const auto numClusters = distributionAttr.num_clusters().getInt();
    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
    auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
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

VPUIP::DistributedBufferType createDMADistributedTensorType(mlir::MLIRContext* ctx, vpux::NDTypeInterface operandType,
                                                            mlir::IntegerAttr numClusters, VPU::ArchKind arch) {
    const auto distMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTiles = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1, numClusters.getInt(), 1});
    const auto heightAlignment = VPU::getSOHMinimalHeightAlignment(operandType.getShape(), numClusters.getInt(), arch);
    const auto alignment =
            heightAlignment == 1 ? nullptr : getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1, heightAlignment, 1});
    const auto uniformDistributedSegments = !VPU::isArchVPUX3XXX(arch) ? mlir::UnitAttr::get(ctx) : nullptr;
    const auto distributionAttr =
            VPU::DistributedTensorAttr::get(distMode, numTiles, nullptr, nullptr, nullptr, numClusters, alignment,
                                            uniformDistributedSegments, nullptr, nullptr, nullptr, ctx);

    const auto memSpace = vpux::IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(ctx, VPU::MemoryKind::CMX_NN));
    const auto order = mlir::AffineMapAttr::get(operandType.getDimsOrder().toAffineMap(ctx));
    auto elemType = operandType.getElementType();

    return VPUIP::DistributedBufferType::get(ctx, operandType.getShape().raw(), elemType, order, memSpace,
                                             distributionAttr);
}

SmallVector<mlir::Operation*> getPureViewLikeOpChains(mlir::Operation* op) {
    VPUX_THROW_UNLESS(op->hasOneUse(), "Op has more than one uses at '{0}'", op->getLoc());
    SmallVector<mlir::Operation*> viewLikeOps;
    auto user = *op->getUsers().begin();
    while (user->hasOneUse()) {
        if (!mlir::isa<VPUIP::GenericReshapeOp, VPUIP::PermuteCastOp, VPUIP::ShapeCastOp>(user) && user->hasOneUse()) {
            break;
        }
        viewLikeOps.push_back(user);
        user = *user->getUsers().begin();
    }
    return viewLikeOps;
}

// check pattern: Copy(ddr->cmx) -> sw.kernel(memPermute)
bool checkPermuteWithoutCopyBackPattern(VPUIP::SwKernelOp swKernelOp, Logger log) {
    if (!VPUIP::isMemPermSwKernel(swKernelOp)) {
        return false;
    }

    log.trace("Got MemPermute SwKernel at {0}. Try to find fuse pattern.", swKernelOp->getLoc());

    if (!VPUIP::isLegalConvertToDMA(swKernelOp, log)) {
        log.nest().trace("VPUIP.SwKernel can not be converted to DMA at {0}", swKernelOp->getLoc());
        return false;
    }

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
    return copyInInputType.getMemoryKind() == VPU::MemoryKind::DDR &&
           copyInOutputType.getMemoryKind() == VPU::MemoryKind::CMX_NN;
}

// check pattern: Copy(ddr->cmx) -> sw.kernel(memPermute) -> Copy (cmx-> ddr)
bool checkPermutePattern(VPUIP::SwKernelOp swKernelOp, Logger log) {
    if (!checkPermuteWithoutCopyBackPattern(swKernelOp, log)) {
        return false;
    }
    if (!swKernelOp->hasOneUse()) {
        log.nest().trace("VPUIP.SwKernel has more than one use at {0}", swKernelOp->getLoc());
        return false;
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    if (copyBackToDDROp == nullptr || !copyBackToDDROp->hasOneUse()) {
        return false;
    }

    const auto copyBackInputType = copyBackToDDROp.input().getType().cast<vpux::NDTypeInterface>();
    const auto copyBackOutputType = copyBackToDDROp.output().getType().cast<vpux::NDTypeInterface>();
    return copyBackInputType.getMemoryKind() == VPU::MemoryKind::CMX_NN &&
           copyBackOutputType.getMemoryKind() == VPU::MemoryKind::DDR;
}

// Check pattern 1: Copy(ddr->cmx) -> sw.kernel(memPermute) -> Copy (cmx-> ddr) -> Copy (ddr-> cmx)
// Check pattern 2: Copy(ddr->cmx) -> sw.kernel(memPermute) -> Copy (cmx-> ddr) -> ClusterCopy (ddr-> cmx)
bool checkPermuteWithCopyPattern(VPUIP::SwKernelOp swKernelOp, Logger log) {
    if (!checkPermutePattern(swKernelOp, log)) {
        return false;
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
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
        auto permuteInType = swKernelOp.getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
        auto permuteOutType = swKernelOp.getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();

        auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp).getValue();
        if (memPerm == DimsOrder::NWHC.toAffineMap(swKernelOp->getContext())) {
            log.trace("MemPermute '{0}' can not be converted to PermuteDMAOp", memPerm);
            return false;
        }
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

// Check pattern: TilingCopy(CMX->DDR) -> Copy(DDR->CMX) -> sw.kernel(memPermute)
bool checkTilingCopyWithPermutePattern(VPUIP::SwKernelOp swKernelOp, Logger log) {
    log.trace("Got sw kernel op at {0}. Try to find permute pattern.", swKernelOp->getLoc());
    if (!checkPermuteWithoutCopyBackPattern(swKernelOp, log)) {
        return false;
    }

    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    auto tilingInputCopy = copyInCMXOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
    if (tilingInputCopy == nullptr || !mlir::isa<VPUIP::CopyOp>(tilingInputCopy.getInnerTaskOp()) ||
        !tilingInputCopy->hasOneUse()) {
        return false;
    }

    auto inDistributedType = tilingInputCopy->getOperand(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (inDistributedType == nullptr) {
        return false;
    }

    const auto inReqs = StrideReqs::compact(inDistributedType.getRank());
    if (!inReqs.checkStrides(inDistributedType)) {
        log.trace("Skip complex case: input is strided");
        return false;
    }

    auto inMode = inDistributedType.getDistribution().mode().getValue();
    return VPU::bitEnumContains(inMode, VPU::DistributionMode::DUPLICATED);
}

bool onlyExpandAtChannel(VPUIP::ExpandOp expandOp) {
    const auto padsBegin = parseIntArrayAttr<int64_t>(expandOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(expandOp.pads_end());

    if (padsBegin.size() != 4 || padsEnd.size() != 4) {
        return false;
    }

    const auto padValues = zip(padsBegin, padsEnd);
    for (auto padValue : padValues | indexed) {
        if (std::get<0>(padValue.value()) != 0 ||
            (std::get<1>(padValue.value()) != 0 && Dim(padValue.index()) != Dims4D::Act::C)) {
            return false;
        }
    }

    return true;
}

bool isExpandOpWrapable(VPUIP::ExpandOp expandOp, Logger log) {
    const auto outputType = expandOp.output().getType().cast<vpux::NDTypeInterface>();

    if (outputType.getDimsOrder() != DimsOrder::NCHW && outputType.getDimsOrder() != DimsOrder::NHWC) {
        log.nest().trace("ExpandOp convert to DMA should have NCHW or NHWC layout.");
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

// Check pattern 1: Expand(ddr->ddr) -> Copy (ddr->cmx) (U8 precision)
// Check pattern 2: Expand(ddr->ddr) -> ClusterCopy (ddr->cmx) (U8 precision)
bool checkExpandU8Pattern(VPUIP::ExpandOp expandOp, Logger log) {
    log.trace("Got ExpandOp at {0}. Try to find fuse pattern.", expandOp->getLoc());

    /*The expandOp was inserted because align to 16 on channel dim. So the expand data is useless. We can fill any data
     * to the expand data. But for convolution, when expand with FP16 precision the expand data will affect the
     * calculation results, if we fill unnormal data like null. For U8 precision any value is a normal data, so we can
     * fill any data to the expand data*/
    const auto outputType = expandOp.output().getType().cast<vpux::NDTypeInterface>();
    if (!outputType.getElementType().isa<mlir::quant::QuantizedType>()) {
        log.nest().trace("ExpandOp convert to DMA shloud with U8 precision.");
        return false;
    }

    return isExpandOpWrapable(expandOp, log);
}

bool checkExpandFP16Pattern(VPUIP::ExpandOp expandOp, Logger log) {
    log.trace("Got ExpandOpFP16 at {0}. Try to find fuse pattern.", expandOp->getLoc());

    /*The expandOp was inserted because align to 16 on channel dim. So the expand data is useless. We can fill any data
     * to the expand data. But for convolution, when expand with FP16 precision the expand data will affect the
     * calculation results, if we fill unnormal data like null.*/

    VPUIP::NCEClusterTaskOp nceTask;
    if (auto expandCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(expandOp->getUsers().begin()))) {
        nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(*(expandCopyOutOp->getUsers().begin()));
    } else if (auto clusterCopyOutOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*(expandOp->getUsers().begin()))) {
        auto innerCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(clusterCopyOutOp.getInnerTaskOp());

        if (innerCopyOp == nullptr) {
            return false;
        }

        auto clusterNCETaskOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*(clusterCopyOutOp->getUsers().begin()));

        if (clusterNCETaskOp == nullptr) {
            return false;
        }

        nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(clusterNCETaskOp.getInnerTaskOp());
    } else {
        return false;
    }

    if (nceTask == nullptr) {
        return false;
    }

    if (nceTask.task_type() == VPUIP::NCETaskType::CONV) {
        return false;
    }

    return isExpandOpWrapable(expandOp, log);
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

    if (!checkExpandU8Pattern(expandOp, log)) {
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

    return checkPermuteWithCopyPattern(swKernelOp, log);
}

// Check Pattern: SW.Kernel(SpaceToDepth) -> Copy(CMX->DDR) -> Copy(DDR->CMX) -> SW.Kernel(MemPermute(Reorder))
bool checkSpaceToDepthWithPermutePattern(VPUIP::SwKernelOp s2dSwKernelOp, Logger log) {
    log.trace("Checking SpaceToDepthWithPermute pattern.");

    if (!VPUIP::isSpaceToDepthSwKernel(s2dSwKernelOp)) {
        log.nest().trace("SWKernel is not SpaceToDepth.");
        return false;
    }

    log.nest().trace("Got SpaceToDepth SwKernel '{0}' at '{1}'.", s2dSwKernelOp->getName(), s2dSwKernelOp->getLoc());

    if (!s2dSwKernelOp->hasOneUse()) {
        log.nest().trace("SpaceToDepth SwKernel should have exactly one use.");
        return false;
    }

    auto copyToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(s2dSwKernelOp->getUsers().begin()));
    if (copyToDDROp == nullptr || !copyToDDROp->hasOneUse()) {
        log.nest().trace("No copy to DDR after SpaceToDepth or copy has not exactly one use.");
        return false;
    }

    const auto copyToDDRInType = copyToDDROp.input().getType().cast<vpux::NDTypeInterface>();
    const auto copyToDDROutType = copyToDDROp.output().getType().cast<vpux::NDTypeInterface>();
    if (copyToDDRInType.getMemoryKind() != VPU::MemoryKind::CMX_NN ||
        copyToDDROutType.getMemoryKind() != VPU::MemoryKind::DDR) {
        log.nest().trace("Copy after SpaceToDepth is not from CMX to DDR.");
        return false;
    }

    auto copyToCMXOp = mlir::dyn_cast<VPUIP::CopyOp>(*(copyToDDROp->getUsers().begin()));
    if (copyToCMXOp == nullptr || !copyToCMXOp->hasOneUse()) {
        log.nest().trace("No copy back to CMX after copy to DDR or copy has not exactly one use.");
        return false;
    }

    const auto copyToCMXInType = copyToCMXOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto copyToCMXOutType = copyToCMXOp.output().getType().cast<vpux::NDTypeInterface>();
    if (copyToCMXInType.getMemoryKind() != VPU::MemoryKind::DDR ||
        copyToCMXOutType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
        log.nest().trace("Copy back to CMX is not from DDR to CMX.");
        return false;
    }

    auto permuteSwKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(*(copyToCMXOp->getUsers().begin()));
    if (permuteSwKernelOp == nullptr) {
        log.nest().trace("No permute found.");
        return false;
    }

    if (!VPUIP::isMemPermSwKernel(permuteSwKernelOp)) {
        log.nest().trace("SWKernel is not MemPermute.");
        return false;
    }

    log.nest().trace("Got MemPermute SWKernel '{0}' at '{1}'.", permuteSwKernelOp->getName(),
                     permuteSwKernelOp->getLoc());

    const auto permuteInOrder = DimsOrder::fromValue(permuteSwKernelOp.inputs().front());
    const auto permuteOutOrder = DimsOrder::fromValue(permuteSwKernelOp.getOutputs().front());
    const auto permuteMemPerm = VPUIP::getMemPermFromSwKernel(permuteSwKernelOp).getValue();
    const auto layoutReorderMemPerm =
            getPermutationFromOrders(permuteInOrder, permuteOutOrder, permuteSwKernelOp.getContext());

    // Only if mem_perm is the same as calculated from in/out orders,
    // we can take it as an layout reorder and merge it into SpaceToDepthDMA
    if (layoutReorderMemPerm != permuteMemPerm) {
        log.nest().trace("MemPermute at '{0}' does not act as an layout reorder.", permuteSwKernelOp->getLoc());
        return false;
    }

    return true;
}

// Check pattern: sw.kernel(spaceToDepth) -> Copy (cmx-> ddr) -> ClusterCopy (ddr-> cmx)
bool checkSpaceToDepthPattern(VPUIP::SwKernelOp swKernelOp, Logger log) {
    if (!VPUIP::isSpaceToDepthSwKernel(swKernelOp)) {
        return false;
    }

    log.trace("Got SpaceToDepth SwKernel at {0}. Try to find fuse pattern.", swKernelOp->getLoc());

    if (!VPUIP::isLegalConvertToDMA(swKernelOp, log)) {
        log.nest().trace("VPUIP.SwKernel can not be converted to DMA at {0}", swKernelOp->getLoc());
        return false;
    }

    auto s2dInType = swKernelOp.getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
    auto s2dOutType = swKernelOp.getResult(0).getType().dyn_cast<vpux::NDTypeInterface>();

    if (!swKernelOp->hasOneUse()) {
        log.nest().trace("VPUIP.SwKernel has more than one use at {0}", swKernelOp->getLoc());
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

    auto clusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*(copyBackToDDROp->getUsers().begin()));
    if (clusterCopyOp == nullptr) {
        return false;
    }
    // It is difficult to use the general method to fuse Permute the with next Cluster Copy Op
    // which has the stride. For example, activation with NHWC layout, need tile at Channel.
    // It is necessary to check the split buffer is continuous.
    const auto clusterOutput = *clusterCopyOp.getOutputs().begin();
    const auto clusterOutputType = clusterOutput.getType().cast<vpux::NDTypeInterface>();

    const auto innerOutput = *clusterCopyOp.getInnerOutputs().begin();
    const auto innerOutputType = innerOutput.getType();
    if (!isSplitContinuousBufferType(innerOutputType, clusterOutputType.dyn_cast<VPUIP::DistributedBufferType>())) {
        return false;
    }

    // Only supports BlocksFirst NHWC->NHWC
    auto s2dAttrs = VPUIP::getSpaceToDepthSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(s2dAttrs.hasValue(), "Cannot extract attributes from SpaceToDepth SwKernel '{0}'.",
                      swKernelOp.getLoc());
    auto mode = s2dAttrs.getValue().first.getValue();
    auto inOrder = s2dInType.getDimsOrder();
    auto outOrder = s2dOutType.getDimsOrder();
    if (!(mode == IE::SpaceToDepthMode::BLOCKS_FIRST && inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NHWC)) {
        return false;
    }

    auto childCopyOp = clusterCopyOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
    if (childCopyOp == nullptr) {
        return false;
    }

    const auto childOutputType = childCopyOp.output().getType().cast<vpux::NDTypeInterface>();
    return childOutputType.getMemoryKind() == VPU::MemoryKind::CMX_NN;
}

// Check pattern 1: Copy(ddr->cmx) -> sw.kernel(PerAxisTile) -> Copy (cmx-> ddr) -> Copy (ddr-> cmx)
// Check pattern 2: Copy(ddr->cmx) -> sw.kernel(PerAxisTile) -> Copy (cmx-> ddr) -> ClusterCopy (ddr-> cmx)
bool checkPerAxisTilePattern(VPUIP::SwKernelOp swKernelOp, Logger log) {
    if (!VPUIP::isPerAxisTileSwKernel(swKernelOp)) {
        return false;
    }

    log.trace("Got PerAxisTile SwKernel at {0}. Try to find fuse pattern.", swKernelOp->getLoc());

    if (!VPUIP::isLegalConvertToDMA(swKernelOp, log)) {
        log.nest().trace("VPUIP.SwKernel can not be converted to DMA at {0}", swKernelOp->getLoc());
        return false;
    }

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
        const auto clusterOutput = *clusterOp.getOutputs().begin();
        const auto clusterOutputType = clusterOutput.getType().cast<vpux::NDTypeInterface>();

        const auto innerOutput = *clusterOp.getInnerOutputs().begin();
        const auto innerOutputType = innerOutput.getType();
        const auto distributedType = clusterOutputType.dyn_cast<VPUIP::DistributedBufferType>();
        VPUX_THROW_UNLESS(distributedType != nullptr, "Can not get distributed type of Cluster Copy Op");
        if (!isSplitContinuousBufferType(innerOutputType, distributedType)) {
            return false;
        }

        const auto perAxisAttrs = VPUIP::getPerAxisTileSwKernelAttr(swKernelOp);
        VPUX_THROW_UNLESS(perAxisAttrs.hasValue(), "Can not get PerAxisTile attribution");
        const auto repeateAxis = perAxisAttrs.getValue().axis;

        // If PerAxisTile Op repeate Axis same with Cluster Copy Tiling Axis
        // Should not fuse PerAxisTileDMA with Cluster Copy Op
        const auto distributionAttr = distributedType.getDistribution();
        if (distributionAttr.num_tiles() != nullptr) {
            const auto isValidTile = [](auto dim) {
                return dim > 1;
            };

            const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());
            const auto iter = llvm::find_if(numTiles, isValidTile);
            if (iter != numTiles.end()) {
                const auto tilingAxis = std::distance(numTiles.begin(), iter);
                if (repeateAxis.getInt() == tilingAxis) {
                    return false;
                }
            }
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
    if (!checkPermuteWithCopyPattern(swKernelOp, _log)) {
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
    if (memPerm == DimsOrder::NWHC.toAffineMap(rewriter.getContext())) {
        _log.trace("MemPermute '{0}' can not be converted to PermuteDMAOp", memPerm);
        return mlir::failure();
    }
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
    if (!checkPermuteWithCopyPattern(swKernelOp, _log)) {
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
    if (memPerm == DimsOrder::NWHC.toAffineMap(rewriter.getContext())) {
        _log.trace("MemPermute '{0}' can not be converted to PermuteDMAOp", memPerm);
        return mlir::failure();
    }

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
    if (!checkExpandU8Pattern(expandOp, _log) && !checkExpandFP16Pattern(expandOp, _log)) {
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
                                           expandOp.pads_beginAttr(), expandOp.pads_endAttr(), nullptr);
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
    if (!checkExpandU8Pattern(expandOp, _log) && !checkExpandFP16Pattern(expandOp, _log)) {
        return mlir::failure();
    }

    _log.trace("Got Expand -> Copy pattern. Expand '{0}' at '{1}'", expandOp->getName(), expandOp->getLoc());

    auto expandCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(expandOp->getUsers().begin()));
    if (expandCopyOutOp == nullptr) {
        return mlir::failure();
    }

    rewriter.setInsertionPointAfter(expandCopyOutOp);
    rewriter.replaceOpWithNewOp<VPUIP::ExpandDMAOp>(expandCopyOutOp, expandOp.input(), expandCopyOutOp.output_buff(),
                                                    expandOp.pads_beginAttr(), expandOp.pads_endAttr(), nullptr);

    _log.nest().trace("Wrap Expand '{0}' at '{1}' with next Copy.", expandOp->getName(), expandOp->getLoc());

    rewriter.eraseOp(expandOp);

    return mlir::success();
}

//
// FusePerAxisTileWithClusterCopy
//

// Copy(ddr->cmx)
//      |
// SW.kernel(PerAxisTile)
//      |                                 ->        Clustering Tiling PerAxisTileDMA (ddr->cmx)
// Copy (cmx->ddr)
//      |
// Clustering Ciling Copy (ddr->cmx)

class FusePerAxisTileWithClusterCopy final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FusePerAxisTileWithClusterCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FusePerAxisTileWithClusterCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swkernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FusePerAxisTileWithClusterCopy::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    if (!checkPerAxisTilePattern(swKernelOp, _log)) {
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

    _log.trace("Got PerAxisTile -> Cluster Copy pattern. PerAxisTile '{0}' at '{1}'", swKernelOp->getName(),
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

    auto perAxisTileAttrs = VPUIP::getPerAxisTileSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(perAxisTileAttrs.hasValue(),
                      "Cannot extract PerAxisTile attribute from perAxisTile SwKernel '{0}'.", swKernelOp.getLoc());

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::PerAxisTileDMAOp>(swKernelOp->getLoc(), newOperands[0], newOperands[1],
                                                perAxisTileAttrs.getValue().axis, perAxisTileAttrs.getValue().repeats,
                                                nullptr);
    };

    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();

    SmallVector<mlir::Value> newNceClusterTilingOperands;
    newNceClusterTilingOperands.push_back(copyInCMXOp.input());
    newNceClusterTilingOperands.push_back(clusterCopyOp.output_buffs()[0]);

    rewriter.setInsertionPointAfter(clusterCopyOp);
    rewriter.replaceOpWithNewOp<VPUIP::NCEClusterTilingOp>(clusterCopyOp, clusterOutputType,
                                                           newNceClusterTilingOperands, bodyBuilder);

    _log.nest().trace("Wrap PerAxisTile '{0}' at '{1}' with next Cluster Copy.", swKernelOp->getName(),
                      swKernelOp->getLoc());

    rewriter.eraseOp(copyBackToDDROp);
    rewriter.eraseOp(swKernelOp);
    rewriter.eraseOp(copyInCMXOp);

    return mlir::success();
}

//
// FuseExpandWithUpsampling
//

class FuseExpandWithUpsampling final : public mlir::OpRewritePattern<VPUIP::ExpandOp> {
public:
    FuseExpandWithUpsampling(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ExpandOp>(ctx), _log(log) {
        setDebugName("FuseExpandWithUpsampling");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseExpandWithUpsampling::matchAndRewrite(VPUIP::ExpandOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Found ExpandOp Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto upsamplingOp = origOp.input().getDefiningOp<VPUIP::UpsamplingUPAOp>();

    if (!upsamplingOp) {
        return mlir::failure();
    }

    if (!onlyExpandAtChannel(origOp)) {
        return mlir::failure();
    }

    _log.trace("Found ExpandOp Operation '{0}' at '{1}'", upsamplingOp->getName(), upsamplingOp->getLoc());
    const auto padChannel = parseIntArrayAttr<int64_t>(origOp.pads_end());
    auto padChannelAttr = getIntArrayAttr(upsamplingOp.getContext(), padChannel);

    const auto outputShape = getShape(origOp.output());

    const auto upsamplingFactorVectorTmp = parseIntArrayAttr<int64_t>(upsamplingOp.upsampling_factor());
    SmallVector<int64_t> upsamplingFactorVector = {1, upsamplingFactorVectorTmp[2], upsamplingFactorVectorTmp[1],
                                                   upsamplingFactorVectorTmp[0]};

    auto constZeros = VPU::getZerosConst(rewriter, outputShape.toValues(), upsamplingOp.input(), origOp.getLoc());

    auto copyZeroOp = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), constZeros, origOp.output_buff());
    auto upsampleFactorAttr = getIntArrayAttr(origOp.getContext(), upsamplingFactorVector);

    auto upsampeDMA = rewriter.replaceOpWithNewOp<VPUIP::UpsamplingDMAOp>(
            origOp, upsamplingOp.input(), copyZeroOp.output(), upsampleFactorAttr, nullptr, padChannelAttr,
            getIntAttr(origOp->getContext(), 0), nullptr, nullptr);

    upsamplingOp.erase();

    _log.trace("Create new upsampling operation '{0}'", upsampeDMA);
    return mlir::success();
}

//
// FusePerAxisTileWithCopy
//

// Copy(ddr->cmx)
//      |
// SW.kernel(PerAxisTile)
//      |                      ->     VPUIP.PerAxisTileDMA(ddr->cmx)
// Copy (cmx->ddr)
//      |
// Copy (ddr->cmx)

class FusePerAxisTileWithCopy final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FusePerAxisTileWithCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FusePerAxisTileWithCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swkernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FusePerAxisTileWithCopy::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                             mlir::PatternRewriter& rewriter) const {
    if (!checkPerAxisTilePattern(swKernelOp, _log)) {
        return mlir::failure();
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    auto childCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*(copyBackToDDROp->getUsers().begin()));
    if (childCopyOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got PerAxisTile -> Copy pattern. PerAxisTile '{0}' at '{1}'", swKernelOp->getName(),
               swKernelOp->getLoc());

    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    auto perAxisTileAttrs = VPUIP::getPerAxisTileSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(perAxisTileAttrs.hasValue(),
                      "Cannot extract perAxisTile attribute from perAxisTile SwKernel '{0}'.", swKernelOp.getLoc());

    rewriter.setInsertionPointAfter(childCopyOp);
    rewriter.replaceOpWithNewOp<VPUIP::PerAxisTileDMAOp>(childCopyOp, copyInCMXOp.input(), childCopyOp.output_buff(),
                                                         perAxisTileAttrs.getValue().axis,
                                                         perAxisTileAttrs.getValue().repeats, nullptr);

    _log.nest().trace("Wrap PerAxisTile '{0}' at '{1}' with next copy.", swKernelOp->getName(), swKernelOp->getLoc());

    rewriter.eraseOp(copyBackToDDROp);
    rewriter.eraseOp(swKernelOp);
    rewriter.eraseOp(copyInCMXOp);

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
    if (memPerm == DimsOrder::NWHC.toAffineMap(rewriter.getContext())) {
        _log.trace("MemPermute '{0}' can not be converted to PermuteDMAOp", memPerm);
        return mlir::failure();
    }

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
    if (memPerm == DimsOrder::NWHC.toAffineMap(rewriter.getContext())) {
        _log.trace("MemPermute '{0}' can not be converted to PermuteDMAOp", memPerm);
        return mlir::failure();
    }

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
// FuseSpaceToDepthAndPermute
//

// SW.Kernel(SpaceToDepth, Layout0->Layout1)
//      |
// Copy(cmx->ddr)
//      |                                             ->      SpaceToDepthDMA(Layout0->Layout2)
// Copy(ddr->cmx)
//      |
// SW.Kernel(MemPermute(Reorder), Layout1->Layout2)

class FuseSpaceToDepthAndPermute final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FuseSpaceToDepthAndPermute(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FuseSpaceToDepthAndPermute");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseSpaceToDepthAndPermute::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Got SWKernel '{0}' at '{1}'.", swKernelOp->getName(), swKernelOp->getLoc());
    if (!checkSpaceToDepthWithPermutePattern(swKernelOp, _log.nest())) {
        _log.nest().trace("Fuse SpaceToDepth and Permute pattern mismatch.");
        return mlir::failure();
    }

    auto s2dCopyOutOp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    auto permuteCopyInOp = mlir::dyn_cast<VPUIP::CopyOp>(*(s2dCopyOutOp->getUsers().begin()));
    auto permuteSwKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(*(permuteCopyInOp->getUsers().begin()));

    const auto s2dInType = swKernelOp.inputs().front().getType().cast<vpux::NDTypeInterface>();
    const auto s2dOutType = swKernelOp.getOutputs().front().getType().cast<vpux::NDTypeInterface>();
    const auto permuteInType = permuteSwKernelOp.inputs().front().getType().cast<vpux::NDTypeInterface>();
    const auto permuteOutType = permuteSwKernelOp.getOutputs().front().getType().cast<vpux::NDTypeInterface>();

    const auto inOrder = s2dInType.getDimsOrder();
    const auto outOrder = permuteOutType.getDimsOrder();

    if (!(inOrder == DimsOrder::NCHW && outOrder == DimsOrder::NHWC)) {
        _log.nest().trace("SpaceToDepthDMA do not support layout '{0}'->'{1}'", inOrder, outOrder);
        return mlir::failure();
    }

    auto spaceToDepthAttrs = VPUIP::getSpaceToDepthSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(spaceToDepthAttrs.hasValue(),
                      "Cannot extract SpaceToDepth attributes from SpaceToDepth SwKernel '{0}'.", swKernelOp.getLoc());
    auto modeAttr = spaceToDepthAttrs.getValue().first;
    auto blockSizeAttr = spaceToDepthAttrs.getValue().second;

    _log.nest().trace("Wrap SpaceToDepth('{0}'->'{1}') and MemPermute('{2}'->'{3}') as SpaceToDepthDMA('{4}'->'{5}')",
                      s2dInType.getDimsOrder(), s2dOutType.getDimsOrder(), permuteInType.getDimsOrder(),
                      permuteOutType.getDimsOrder(), inOrder, outOrder);

    auto input = swKernelOp.getOperand(0);

    auto outputMemRef = permuteOutType.cast<mlir::MemRefType>();
    auto allocSpaceToDepthOp = rewriter.create<mlir::memref::AllocOp>(permuteSwKernelOp->getLoc(), outputMemRef);

    rewriter.replaceOpWithNewOp<VPUIP::SpaceToDepthDMAOp>(permuteSwKernelOp, input, allocSpaceToDepthOp, blockSizeAttr,
                                                          modeAttr, nullptr);

    rewriter.eraseOp(permuteCopyInOp);
    rewriter.eraseOp(s2dCopyOutOp);
    rewriter.eraseOp(swKernelOp);

    return mlir::success();
}

//
// WrapDepthToSpaceAsClusterNNDMA
//

// Match this pattern to convert SWKernel DepthToSpace to
// multi-cluster DepthToSpaceDMA.
//
//   --- (Optional if no cluster output) ---
//   |        ClusterCopy(cmx->ddr)        |
//   |                 |                   |
//   |           Copy(ddr->cmx)            |
//   -----------       |         -----------
//            SWKernel(DepthToSpace)
//                     |
//   ----------- Copy(cmx->ddr)  -----------
//   |                 |                   |
//   |            [ShapeCast]              |
//   |                 |                   |
//   |        ClusterCopy(ddr->cmx)        |
//   --- (Optional if no cluster input)  ---

class WrapDepthToSpaceAsClusterNNDMA final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    WrapDepthToSpaceAsClusterNNDMA(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("WrapDepthToSpaceAsClusterNNDMA");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swkernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    enum PatternType { NONE, INPUT, OUTPUT, BOTH };
    PatternType checkPattern(VPUIP::SwKernelOp swKernelOp) const;
    bool isValidConversion(VPUIP::SwKernelOp swKernelOp) const;

private:
    Logger _log;
};

bool WrapDepthToSpaceAsClusterNNDMA::isValidConversion(VPUIP::SwKernelOp swKernelOp) const {
    _log.trace("Checking DepthToSpaceAsMultiCluster pattern.");

    if (!VPUIP::isDepthToSpaceSwKernel(swKernelOp)) {
        _log.nest().trace("SWKernel is not DepthToSpace.");
        return false;
    }

    _log.nest().trace("Got DepthToSpace SwKernel '{0}' at '{1}'.", swKernelOp->getName(), swKernelOp->getLoc());

    if (!VPUIP::isLegalConvertToDMA(swKernelOp, _log.nest())) {
        _log.nest().trace("VPUIP.SwKernel can not be converted to DMA at {0}", swKernelOp->getLoc());
        return false;
    }

    const auto d2sInType = swKernelOp.inputs().front().getType().cast<vpux::NDTypeInterface>();
    const auto d2sOutType = swKernelOp.getOutputs().front().getType().cast<vpux::NDTypeInterface>();
    const auto inOrder = d2sInType.getDimsOrder();
    const auto outOrder = d2sOutType.getDimsOrder();
    if (inOrder != DimsOrder::NHWC || outOrder != DimsOrder::NHWC) {
        _log.nest().trace("Only support NHWC->NHWC, but got: '{0}'->'{1}'", inOrder, outOrder);
        return false;
    }

    auto d2sAttrs = VPUIP::getDepthToSpaceSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(d2sAttrs.hasValue(), "Cannot extract DepthToSpace attributes from SwKernel '{0}'.",
                      swKernelOp.getLoc());
    auto mode = std::get<0>(d2sAttrs.getValue()).getValue();
    if (mode != IE::DepthToSpaceMode::BLOCKS_FIRST) {
        _log.nest().trace("Only support BlocksFirst mode");
        return false;
    }
    return true;
}

WrapDepthToSpaceAsClusterNNDMA::PatternType WrapDepthToSpaceAsClusterNNDMA::checkPattern(
        VPUIP::SwKernelOp swKernelOp) const {
    if (!isValidConversion(swKernelOp)) {
        return PatternType::NONE;
    }

    const auto isSegmented = [&](vpux::NDTypeInterface operandType) {
        auto operandDistType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
        VPUX_THROW_WHEN(operandDistType == nullptr, "Operand is not distributed type");
        const auto distribution = operandDistType.getDistribution();
        const auto mode = distribution.mode().getValue();
        return mode == VPU::DistributionMode::SEGMENTED;
    };

    const auto isValidClusterCopyOp = [&](VPUIP::NCEClusterTilingOp clusterOp, bool isCopyIn) {
        auto childCopyOp = clusterOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
        if (childCopyOp == nullptr) {
            return false;
        }
        const auto childCopyType = isCopyIn ? childCopyOp.input().getType().cast<vpux::NDTypeInterface>()
                                            : childCopyOp.output().getType().cast<vpux::NDTypeInterface>();
        if (childCopyType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
            return false;
        }

        const auto clusterOperand = isCopyIn ? *clusterOp.getInputs().begin() : *clusterOp.getOutputs().begin();
        const auto clusterOperandType = clusterOperand.getType().cast<vpux::NDTypeInterface>();
        const auto innerOperand = isCopyIn ? *clusterOp.getInnerInputs().begin() : *clusterOp.getInnerOutputs().begin();
        const auto innerOperandType = innerOperand.getType();
        if (!isSegmented(clusterOperandType) ||
            !isSplitContinuousBufferType(innerOperandType,
                                         clusterOperandType.dyn_cast<VPUIP::DistributedBufferType>())) {
            return false;
        }
        return true;
    };

    const auto isValidCopyOp = [&](VPUIP::CopyOp copyOp, bool isCopyIn) {
        const auto copyInputType = copyOp.input().getType().cast<vpux::NDTypeInterface>();
        const auto copyOutputType = copyOp.output().getType().cast<vpux::NDTypeInterface>();
        return isCopyIn ? (copyInputType.getMemoryKind() == VPU::MemoryKind::DDR &&
                           copyOutputType.getMemoryKind() == VPU::MemoryKind::CMX_NN)
                        : (copyInputType.getMemoryKind() == VPU::MemoryKind::CMX_NN &&
                           copyOutputType.getMemoryKind() == VPU::MemoryKind::DDR);
    };

    const auto findInputClusterCopy = [&](VPUIP::SwKernelOp swKernelOp) -> mlir::Operation* {
        auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
        if (copyInCMXOp == nullptr || !copyInCMXOp->hasOneUse() || !isValidCopyOp(copyInCMXOp, true)) {
            return nullptr;
        }
        const auto copyInInputType = copyInCMXOp.input().getType().cast<vpux::NDTypeInterface>();
        const auto copyInOutputType = copyInCMXOp.output().getType().cast<vpux::NDTypeInterface>();
        if (copyInInputType.getMemoryKind() != VPU::MemoryKind::DDR ||
            copyInOutputType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
            return nullptr;
        }
        auto clusterOp = copyInCMXOp.getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
        if (clusterOp == nullptr || !clusterOp->hasOneUse() || !isValidClusterCopyOp(clusterOp, true)) {
            return nullptr;
        }
        return clusterOp;
    };

    const auto findOutputClusterCopy = [&](VPUIP::SwKernelOp swKernelOp) -> mlir::Operation* {
        if (!swKernelOp->hasOneUse()) {
            return nullptr;
        }
        auto copyOutDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
        if (copyOutDDROp == nullptr || !copyOutDDROp->hasOneUse() || !isValidCopyOp(copyOutDDROp, false)) {
            return nullptr;
        }
        auto potentialViewLikeOp = *copyOutDDROp->getUsers().begin();
        while (VPUIP::isPureViewOp(potentialViewLikeOp)) {
            if (!potentialViewLikeOp->hasOneUse()) {
                return nullptr;
            }
            potentialViewLikeOp = *potentialViewLikeOp->getUsers().begin();
        }
        auto clusterOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(potentialViewLikeOp);
        if (clusterOp == nullptr || !isValidClusterCopyOp(clusterOp, false)) {
            return nullptr;
        }
        return clusterOp;
    };

    const auto clusterCopyIn = findInputClusterCopy(swKernelOp);
    const auto clusterCopyOut = findOutputClusterCopy(swKernelOp);

    if (clusterCopyIn == nullptr && clusterCopyOut == nullptr) {
        _log.nest().trace("Neither input nor output is in multicluster");
        return PatternType::NONE;
    } else if (clusterCopyIn != nullptr && clusterCopyOut == nullptr) {
        _log.nest().trace("Found input in multicluster");
        return PatternType::INPUT;
    } else if (clusterCopyIn == nullptr && clusterCopyOut != nullptr) {
        _log.nest().trace("Found output in multicluster");
        return PatternType::OUTPUT;
    } else {
        _log.nest().trace("Found both input and output in multicluster");
        return PatternType::BOTH;
    }
}

mlir::LogicalResult WrapDepthToSpaceAsClusterNNDMA::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    const auto patternType = checkPattern(swKernelOp);
    if (patternType == PatternType::NONE) {
        return mlir::failure();
    }

    _log.trace("Found DepthToSpace at '{0}' with ClusterCopy pattern", swKernelOp->getLoc());

    auto ctx = swKernelOp.getContext();
    auto arch = VPU::getArch(swKernelOp.getOperation());

    // Extract D2S attributes
    auto d2sAttrs = VPUIP::getDepthToSpaceSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(d2sAttrs.hasValue(), "Failed to extract attributes from DepthToSpace SwKernel '{0}'.",
                      swKernelOp.getLoc());
    auto modeAttr = std::get<0>(d2sAttrs.getValue());
    auto blockSizeAttr = std::get<1>(d2sAttrs.getValue());
    auto paddedChannels = std::get<2>(d2sAttrs.getValue());

    // Extract D2S output
    auto d2sOutputBuff = swKernelOp.getOperand(1);
    auto d2sOutputBuffType = d2sOutputBuff.getType().cast<vpux::NDTypeInterface>();

    // Rewrite subgraph
    const auto copyBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
    };
    const auto d2sBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        builder.create<VPUIP::DepthToSpaceDMAOp>(loc, newOperands[0], newOperands[1], blockSizeAttr, modeAttr, nullptr,
                                                 paddedChannels);
    };

    const auto numClusters = getIntAttr(ctx, VPUIP::getNumClusterUsed(swKernelOp->getParentOfType<mlir::ModuleOp>()));

    const auto insertionPoint = patternType == PatternType::OUTPUT || patternType == PatternType::BOTH
                                        ? *swKernelOp->getUsers().begin()
                                        : swKernelOp.getOperation();
    rewriter.setInsertionPointAfter(insertionPoint);

    SmallVector<mlir::Operation*> opsToErase;

    // If pattern is INPUT or BOTH, which means the input side is in multicluster,
    // we need to create an input cluster copy before ClusterD2SDMAOp
    auto d2sInput = swKernelOp.getOperand(0);
    if (patternType == PatternType::INPUT || patternType == PatternType::BOTH) {
        auto inputCopyOp = d2sInput.getDefiningOp<VPUIP::CopyOp>();
        VPUX_THROW_WHEN(inputCopyOp == nullptr, "Failed to get input copy of DepthToSpace");
        auto clusterCopyInAllocType =
                createDMADistributedTensorType(ctx, inputCopyOp.output().getType(), numClusters, arch);
        auto clusterCopyInAllocOp = rewriter.create<VPURT::AllocDistributed>(inputCopyOp.getLoc(),
                                                                             clusterCopyInAllocType, nullptr, nullptr);
        opsToErase.push_back(inputCopyOp);
        auto d2sCopyInOp = rewriter.create<VPUIP::NCEClusterTilingOp>(
                inputCopyOp.getLoc(), clusterCopyInAllocType,
                SmallVector<mlir::Value>{inputCopyOp.input(), clusterCopyInAllocOp.buffer()}, copyBodyBuilder);
        d2sInput = d2sCopyInOp.getResult(0);
        _log.nest().trace("Create new cluster copy-in op: {0}", d2sCopyInOp);
    }

    // If pattern is OUTPUT or BOTH, which means the output side is in multicluster,
    // we need to create ClusterD2SDMAOp with a following output cluster copy
    // Otherwise, when the pattern is INPUT, which means the output side is not in multicluster,
    // we only need to create a ClusterD2SDMAOp
    // The created ClusterD2SDMAOp will be unrolled according to its input/output type later in
    // UnrollDepthToSpaceDMAPass
    if (patternType == PatternType::OUTPUT || patternType == PatternType::BOTH) {
        auto outputCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*swKernelOp->getUsers().begin());
        VPUX_THROW_WHEN(outputCopyOp == nullptr, "Failed to get output copy of DepthToSpace");
        // create d2s
        auto d2sOutAllocType = createDMADistributedTensorType(ctx, d2sOutputBuffType, numClusters, arch);
        auto d2sOutAllocOp =
                rewriter.create<VPURT::AllocDistributed>(swKernelOp.getLoc(), d2sOutAllocType, nullptr, nullptr);
        auto d2sOp = rewriter.create<VPUIP::NCEClusterTilingOp>(
                swKernelOp.getLoc(), d2sOutAllocType, SmallVector<mlir::Value>{d2sInput, d2sOutAllocOp.buffer()},
                d2sBodyBuilder);
        _log.nest().trace("Create new cluster DepthToSpaceDMAOp: {0}", d2sOp);
        opsToErase.push_back(swKernelOp);
        // create output copy
        auto d2sCopyOutOp = rewriter.replaceOpWithNewOp<VPUIP::NCEClusterTilingOp>(
                outputCopyOp, outputCopyOp.output_buff().getType(),
                SmallVector<mlir::Value>{d2sOp.getResult(0), outputCopyOp.output_buff()}, copyBodyBuilder);
        _log.nest().trace("Create new cluster copy-out op: {0}", d2sCopyOutOp);
    } else {
        // create d2s
        auto d2sOp = rewriter.replaceOpWithNewOp<VPUIP::NCEClusterTilingOp>(
                swKernelOp, d2sOutputBuffType, SmallVector<mlir::Value>{d2sInput, d2sOutputBuff}, d2sBodyBuilder);
        _log.nest().trace("Create new cluster DepthToSpaceDMAOp: {0}", d2sOp);
    }

    while (!opsToErase.empty()) {
        _log.nest().trace("Erase Op: {0}", opsToErase.back()->getLoc());
        rewriter.eraseOp(opsToErase.back());
        opsToErase.pop_back();
    }

    return mlir::success();
}

//
// FuseSpaceToDepthWithClusterCopy
//

// SW.kernel(spaceToDepth)
//      |                                 ->        Clustering Tiling SpaceToDepthDMA (cmx->cmx)
// Copy (cmx->ddr)
//      |
// Clustering Ciling Copy (ddr->cmx)

class FuseSpaceToDepthWithClusterCopy final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FuseSpaceToDepthWithClusterCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FuseSpaceToDepthWithClusterCopy");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swkernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseSpaceToDepthWithClusterCopy::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Got SWKernel '{0}' at '{1}'.", swKernelOp->getName(), swKernelOp->getLoc());
    if (!checkSpaceToDepthPattern(swKernelOp, _log.nest())) {
        return mlir::failure();
    }

    auto copyBackToDDROp = mlir::dyn_cast<VPUIP::CopyOp>(*(swKernelOp->getUsers().begin()));
    auto clusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*(copyBackToDDROp->getUsers().begin()));

    _log.nest().trace("Got SpaceToDepth -> Cluster Copy pattern. SpaceToDepth '{0}' at '{1}'", swKernelOp->getName(),
                      swKernelOp->getLoc());

    // Check distribution mode
    const auto clusterOutput = *clusterCopyOp.getOutputs().begin();
    const auto clusterOutputType = clusterOutput.getType().cast<vpux::NDTypeInterface>();
    const auto distributedType = clusterOutputType.dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedType == nullptr) {
        return mlir::failure();
    }

    auto s2dAttrs = VPUIP::getSpaceToDepthSwKernelAttr(swKernelOp);
    auto modeAttr = s2dAttrs.getValue().first;
    auto blockSizeAttr = s2dAttrs.getValue().second;

    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.mode().getValue();
    const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.num_tiles());

    // Currently only support SOH
    if (mode == VPU::DistributionMode::SEGMENTED && numTiles[Dims4D::Act::H.ind()] > 1) {
        const auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
        // H per cluster must be divisible by block size
        auto isHeightDivisible = [&](ShapeRef shape) {
            return shape[Dims4D::Act::H] % blockSizeAttr.getInt() != 0;
        };
        if (llvm::any_of(perClusterShapes, isHeightDivisible)) {
            return mlir::failure();
        }
    } else {
        return mlir::failure();
    }

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::SpaceToDepthDMAOp>(swKernelOp->getLoc(), newOperands[0], newOperands[1], blockSizeAttr,
                                                 modeAttr, nullptr);
    };

    SmallVector<mlir::Value> newNceClusterTilingOperands;
    newNceClusterTilingOperands.push_back(swKernelOp.getOperand(0));
    newNceClusterTilingOperands.push_back(clusterCopyOp.output_buffs()[0]);

    rewriter.setInsertionPointAfter(clusterCopyOp);
    rewriter.replaceOpWithNewOp<VPUIP::NCEClusterTilingOp>(clusterCopyOp, clusterOutputType,
                                                           newNceClusterTilingOperands, bodyBuilder);

    _log.nest().trace("Wrap SpaceToDepth '{0}' at '{1}' with next Cluster Copy.", swKernelOp->getName(),
                      swKernelOp->getLoc());

    rewriter.eraseOp(copyBackToDDROp);
    rewriter.eraseOp(swKernelOp);

    return mlir::success();
}

//
// FuseClusterCopyWithMemPermute
//

// Duplicated Tiling Copy(cmx->ddr)
//              |
//         Copy(ddr->cmx)                ->        Clustering Tiling PermuteDMA (cmx->cmx)
//              |
//         SW.kernel(memPermute)
//              |

class FuseClusterCopyWithMemPermute final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    FuseClusterCopyWithMemPermute(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("FuseClusterCopyWithMemPermute");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseClusterCopyWithMemPermute::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    if (!checkTilingCopyWithPermutePattern(swKernelOp, _log)) {
        return mlir::failure();
    }
    auto copyInCMXOp = swKernelOp.getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    auto tilingInputCopy = copyInCMXOp->getOperand(0).getDefiningOp<VPUIP::NCEClusterTilingOp>();
    VPUX_THROW_WHEN(copyInCMXOp == nullptr || tilingInputCopy == nullptr, "Invalid copy");

    _log.trace("Process sw kernel op {0}", swKernelOp);

    auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp).getValue();
    if (memPerm == DimsOrder::NWHC.toAffineMap(rewriter.getContext())) {
        _log.trace("MemPermute '{0}' can not be converted to PermuteDMAOp", memPerm);
        return mlir::failure();
    }

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::PermuteDMAOp>(swKernelOp->getLoc(), newOperands[0], newOperands[1],
                                            mlir::AffineMapAttr::get(memPerm), nullptr);
    };

    SmallVector<mlir::Value> newNceClusterTilingOperands;
    newNceClusterTilingOperands.push_back(tilingInputCopy.getOperand(0));
    newNceClusterTilingOperands.push_back(swKernelOp.output_buffs()[0]);

    rewriter.replaceOpWithNewOp<VPUIP::NCEClusterTilingOp>(swKernelOp, swKernelOp.getResult(0).getType(),
                                                           newNceClusterTilingOperands, bodyBuilder);
    rewriter.eraseOp(copyInCMXOp);
    rewriter.eraseOp(tilingInputCopy);

    return mlir::success();
}

//
// FuseClusterMemPermuteWithViewLikeOps
//

//  Cluster Tiling PermuteDMA (cmx->cmx)
//              |
//          ViewLikeOp                          Clustering Tiling PermuteDMA (cmx->cmx)
//              |                       ->                 |
//         Copy (cmx->ddr)                              ViewLikeOp
//              |
// Duplicated Tiling Copy (ddr->cmx)

class FuseClusterMemPermuteWithViewLikeOps final : public mlir::OpRewritePattern<VPUIP::PermuteDMAOp> {
public:
    FuseClusterMemPermuteWithViewLikeOps(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::PermuteDMAOp>(ctx), _log(log) {
        setDebugName("FuseClusterMemPermuteWithViewLikeOps");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::PermuteDMAOp permuteOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseClusterMemPermuteWithViewLikeOps::matchAndRewrite(VPUIP::PermuteDMAOp permuteOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(permuteOp->getParentOp());
    if (clusterTilingOp == nullptr || !clusterTilingOp->hasOneUse()) {
        return mlir::failure();
    }
    auto viewLikeOps = getPureViewLikeOpChains(clusterTilingOp);

    // check copy op after viewLikeOps
    auto userOp = viewLikeOps.empty() ? *clusterTilingOp->getUsers().begin() : *viewLikeOps.back()->getUsers().begin();
    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(userOp);
    if (copyOp == nullptr || !copyOp->hasOneUse()) {
        return mlir::failure();
    }
    const auto copyInType = copyOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto copyOutType = copyOp.output().getType().cast<vpux::NDTypeInterface>();
    if (copyInType.getMemoryKind() != VPU::MemoryKind::CMX_NN || copyOutType.getMemoryKind() != VPU::MemoryKind::DDR) {
        return mlir::failure();
    }

    // check tiling copy op
    auto tilingCopyUserOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(*copyOp->getUsers().begin());
    if (tilingCopyUserOp == nullptr || !mlir::isa<VPUIP::CopyOp>(tilingCopyUserOp.getInnerTaskOp())) {
        return mlir::failure();
    }
    const auto tilingCopyOutType = tilingCopyUserOp->getResult(0).getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto outDistribution = tilingCopyOutType.getDistribution();
    auto mode = outDistribution.mode().getValue();
    if (tilingCopyOutType.getMemoryKind() != VPU::MemoryKind::CMX_NN ||
        !VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED)) {
        return mlir::failure();
    }

    const auto outReqs = StrideReqs::compact(tilingCopyOutType.getRank());
    if (!outReqs.checkStrides(tilingCopyOutType)) {
        _log.trace("Skip complex case: output is strided");
        return mlir::failure();
    }

    // create new Cluster Tiling PermuteDMA with distiributed output
    rewriter.setInsertionPointAfter(tilingCopyUserOp);
    auto memPerm = permuteOp.mem_permAttr();
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
        builder.create<VPUIP::PermuteDMAOp>(permuteOp->getLoc(), newOperands[0], newOperands[1], memPerm, nullptr);
    };
    const auto ctx = permuteOp->getContext();
    const auto outType = permuteOp.output().getType().dyn_cast<vpux::NDTypeInterface>();
    const auto outShape = outType.getShape();
    const auto outElemType = outType.getElementType();
    const auto order = mlir::AffineMapAttr::get(outType.getDimsOrder().toAffineMap(ctx));
    auto newPermuteDistributedOutType = VPUIP::DistributedBufferType::get(
            ctx, outShape.raw(), outElemType, order, tilingCopyOutType.getMemSpace(), outDistribution);

    auto newAlloc = rewriter.create<VPURT::AllocDistributed>(permuteOp->getLoc(), newPermuteDistributedOutType, nullptr,
                                                             nullptr);

    SmallVector<mlir::Value> newNceClusterTilingOperands;
    newNceClusterTilingOperands.push_back(clusterTilingOp.getOperand(0));
    newNceClusterTilingOperands.push_back(newAlloc);
    auto newPermuteOp = rewriter.create<VPUIP::NCEClusterTilingOp>(permuteOp->getLoc(), newPermuteDistributedOutType,
                                                                   newNceClusterTilingOperands, bodyBuilder);
    _log.trace("create new cluster tiling permute op {0}", newPermuteOp);

    // create new view like ops
    auto newOutput = newPermuteOp->getResult(0);
    for (auto viewLikeOp : viewLikeOps) {
        mlir::BlockAndValueMapping mapper;
        mapper.map(viewLikeOp->getOperands(), makeArrayRef({newOutput}));
        auto* newViewLikeOp = rewriter.clone(*viewLikeOp, mapper);

        auto viewLikeOutType = viewLikeOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
        auto viewLikeOutShape = viewLikeOutType.getShape();
        auto viewLikeOutOrder = mlir::AffineMapAttr::get(viewLikeOutType.getDimsOrder().toAffineMap(ctx));
        auto viewLikeElemType = viewLikeOutType.getElementType();
        auto newViewLikeOutType =
                VPUIP::DistributedBufferType::get(ctx, viewLikeOutShape.raw(), viewLikeElemType, viewLikeOutOrder,
                                                  tilingCopyOutType.getMemSpace(), outDistribution);
        newViewLikeOp->getResult(0).setType(newViewLikeOutType);
        newViewLikeOp->dump();
        newOutput = newViewLikeOp->getResult(0);

        viewLikeOp->dropAllUses();
        rewriter.eraseOp(viewLikeOp);
    }
    rewriter.replaceOp(tilingCopyUserOp, newOutput);
    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(clusterTilingOp);
    return mlir::success();
}

//
// WrapWithPermuteAsNNDMAPass
//

class WrapWithPermuteAsNNDMAPass final : public VPUIP::WrapWithPermuteAsNNDMABase<WrapWithPermuteAsNNDMAPass> {
public:
    explicit WrapWithPermuteAsNNDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

// TODO: #71565
void WrapWithPermuteAsNNDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseExpandAndPermuteWithClusterCopy>(&ctx, _log);
    patterns.add<FuseExpandAndPermuteWithCopy>(&ctx, _log);
    patterns.add<FuseExpandWithClusterCopy>(&ctx, _log);
    patterns.add<FuseExpandWithCopy>(&ctx, _log);
    patterns.add<FuseExpandWithUpsampling>(&ctx, _log);
    patterns.add<FuseMemPermuteWithClusterCopy>(&ctx, _log);
    patterns.add<FuseMemPermuteWithCopy>(&ctx, _log);
    patterns.add<FuseClusterCopyWithMemPermute>(&ctx, _log);
    patterns.add<FuseClusterMemPermuteWithViewLikeOps>(&ctx, _log);
    patterns.add<FusePerAxisTileWithClusterCopy>(&ctx, _log);
    patterns.add<FusePerAxisTileWithCopy>(&ctx, _log);
    patterns.add<FuseSpaceToDepthAndPermute>(&ctx, _log);
    patterns.add<FuseSpaceToDepthWithClusterCopy>(&ctx, _log);
    patterns.add<WrapDepthToSpaceAsClusterNNDMA>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createWrapWithPermuteAsNNDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createWrapWithPermuteAsNNDMAPass(Logger log) {
    return std::make_unique<WrapWithPermuteAsNNDMAPass>(log);
}

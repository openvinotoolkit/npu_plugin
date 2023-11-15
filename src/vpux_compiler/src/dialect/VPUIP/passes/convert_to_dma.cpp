//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;
namespace {

//
// ConvertToDMAPass
//

class ConvertToDMAPass final : public VPUIP::ConvertToDMABase<ConvertToDMAPass> {
public:
    explicit ConvertToDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void getDependentDialects(mlir::DialectRegistry& registry) const override {
        registry.insert<vpux::VPUIP::VPUIPDialect>();
    }

public:
    class DepthToSpaceConverter;
    class MemPermuteConverter;
    class SpaceToDepthConverter;
    class ExpandConverter;
    class PerAxisTileConverter;
    class SwKernelMemPermuteConverter;
    class SwKernelDepthToSpaceConverter;
    class SwKernelSpaceToDepthConverter;
    class SwKernelPerAxisTileConverter;
    class UpsamplingOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// DepthToSpaceConverter
//

class ConvertToDMAPass::DepthToSpaceConverter final : public mlir::OpRewritePattern<VPUIP::DepthToSpaceUPAOp> {
public:
    DepthToSpaceConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::DepthToSpaceUPAOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::DepthToSpaceConverter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::DepthToSpaceUPAOp depthToSpaceOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::DepthToSpaceConverter::matchAndRewrite(VPUIP::DepthToSpaceUPAOp depthToSpaceOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("DepthtoSpace rewriter operation '{0}' at '{1}'", depthToSpaceOp->getName(), depthToSpaceOp->getLoc());

    // insert copy before DepthToSpace
    auto memRefInputType = depthToSpaceOp.input().getType().cast<mlir::MemRefType>();
    auto cmxIndexSymbolAttr = IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
    auto newMemRefInputType = mlir::MemRefType::get(memRefInputType.getShape(), memRefInputType.getElementType(),
                                                    memRefInputType.getLayout(), cmxIndexSymbolAttr);
    auto allocInputOp = rewriter.create<mlir::memref::AllocOp>(depthToSpaceOp->getLoc(), newMemRefInputType);
    auto inputCopyOp = rewriter.create<VPUIP::CopyOp>(depthToSpaceOp->getLoc(), depthToSpaceOp.input(), allocInputOp);
    _log.trace("Insert copy Op before DepthToSpaceAsDMA Op with alloc buffer location {0}.",
               newMemRefInputType.getMemorySpace());

    // create new DepthToSpaceAsDMA Op
    auto depthToSpaceMemRefType = depthToSpaceOp.getType();
    auto newDepthToSpaceMemRefType =
            mlir::MemRefType::get(depthToSpaceMemRefType.getShape(), depthToSpaceMemRefType.getElementType(),
                                  depthToSpaceMemRefType.getLayout(), cmxIndexSymbolAttr);
    auto allocDepthToSpaceOp =
            rewriter.create<mlir::memref::AllocOp>(depthToSpaceOp->getLoc(), newDepthToSpaceMemRefType);
    auto newDepthToSpaceOp = rewriter.create<VPUIP::DepthToSpaceDMAOp>(
            depthToSpaceOp->getLoc(), inputCopyOp.output(), allocDepthToSpaceOp, depthToSpaceOp.block_sizeAttr(),
            depthToSpaceOp.modeAttr(), nullptr, depthToSpaceOp.padded_channelsAttr());
    _log.trace("Create new DepthToSpaceAsDMA Op with alloc buffer location {0}.",
               newDepthToSpaceMemRefType.getMemorySpace());

    // create copy after DepthToSpace
    auto memRefOutputType = depthToSpaceOp.output().getType().cast<mlir::MemRefType>();
    auto newMemRefOuputType =
            mlir::MemRefType::get(memRefOutputType.getShape(), memRefOutputType.getElementType(),
                                  memRefOutputType.getLayout(), IndexedSymbolAttr::get(rewriter.getContext(), "DDR"));
    auto allocOutputOp = rewriter.create<mlir::memref::AllocOp>(depthToSpaceOp->getLoc(), newMemRefOuputType);
    rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(depthToSpaceOp, newDepthToSpaceOp->getResult(0), allocOutputOp);
    _log.trace("Insert copy Op after DepthToSpaceAsDMA Op with alloc buffer location {0}.",
               newMemRefOuputType.getMemorySpace());

    return mlir::success();
}

//
// MemPermuteConverter
//

class ConvertToDMAPass::MemPermuteConverter final : public mlir::OpRewritePattern<VPUIP::PermuteUPAOp> {
public:
    MemPermuteConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::PermuteUPAOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::MemPermuteConverter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::PermuteUPAOp permuteUPAOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::MemPermuteConverter::matchAndRewrite(VPUIP::PermuteUPAOp permuteUPAOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("MemPermute rewriter operation '{0}' at '{1}'", permuteUPAOp->getName(), permuteUPAOp->getLoc());

    // insert copy before MemPermuteOp
    auto memRefInputType = permuteUPAOp.input().getType().cast<mlir::MemRefType>();
    auto newMemRefInputType = mlir::MemRefType::get(
            memRefInputType.getShape(), memRefInputType.getElementType(), memRefInputType.getLayout(),
            IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0));
    auto allocInputOp = rewriter.create<mlir::memref::AllocOp>(permuteUPAOp->getLoc(), newMemRefInputType);
    auto inputCopyOp = rewriter.create<VPUIP::CopyOp>(permuteUPAOp->getLoc(), permuteUPAOp.input(), allocInputOp);
    _log.trace("Insert copy Op before MemPermute Op with alloc buffer location {0}.",
               newMemRefInputType.getMemorySpace());

    // create new MemPermuteOp
    auto permuteMemRefType = permuteUPAOp.getType().dyn_cast<mlir::MemRefType>();
    VPUX_THROW_WHEN(permuteMemRefType == nullptr, "Unexpected output type for VPUIP.PermuteUPA at '{0}'",
                    permuteUPAOp.getLoc());
    auto newPermuteMemRefType = mlir::MemRefType::get(
            permuteMemRefType.getShape(), permuteMemRefType.getElementType(), permuteMemRefType.getLayout(),
            IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0));
    auto allocPermuteOp = rewriter.create<mlir::memref::AllocOp>(permuteUPAOp->getLoc(), newPermuteMemRefType);
    auto newMemPermuteOp = rewriter.create<VPUIP::PermuteDMAOp>(
            permuteUPAOp->getLoc(), inputCopyOp.output(), allocPermuteOp, permuteUPAOp.order_valueAttr(), nullptr);
    _log.trace("Create new PermuteDMA Op with alloc buffer location {0}.", newPermuteMemRefType.getMemorySpace());

    // create copy after MemPermuteDMAOp
    auto getOutputAllocOp = [&]() -> mlir::Value {
        // The input of child of permuteUPAOp should be checked as well as the output,
        // because the input of child could be a BlockArgument while the output of permuteUPAOp is not
        if (permuteUPAOp.output_buff().isa<mlir::BlockArgument>()) {
            _log.trace("Insert copy Op after MemPermute Op with alloc buffer goes to output.");
            return permuteUPAOp.output_buff();
        } else if (permuteUPAOp.output().use_empty()) {
            _log.trace("MemPermute Op is the last Op outputs to a viewlike Op(QuantizeCast/ PermuteCast)");
            return permuteUPAOp.output_buff();
        }

        auto memRefOutputType = permuteUPAOp.output().getType().cast<mlir::MemRefType>();
        auto newMemRefOuputType = mlir::MemRefType::get(memRefOutputType.getShape(), memRefOutputType.getElementType(),
                                                        memRefOutputType.getLayout(),
                                                        IndexedSymbolAttr::get(rewriter.getContext(), "DDR"));
        _log.trace("Insert copy Op after MemPermute Op with alloc buffer location {0}.",
                   newMemRefOuputType.getMemorySpace());
        return rewriter.create<mlir::memref::AllocOp>(permuteUPAOp->getLoc(), newMemRefOuputType);
    };

    rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(permuteUPAOp, newMemPermuteOp->getResult(0), getOutputAllocOp());

    return mlir::success();
}

//
// SpaceToDepthConverter
//

class ConvertToDMAPass::SpaceToDepthConverter final : public mlir::OpRewritePattern<VPUIP::SpaceToDepthUPAOp> {
public:
    SpaceToDepthConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SpaceToDepthUPAOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::SpaceToDepthConverter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SpaceToDepthUPAOp spaceToDepthOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::SpaceToDepthConverter::matchAndRewrite(VPUIP::SpaceToDepthUPAOp spaceToDepthOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("SpaceToDepth rewriter operation '{0}' at '{1}'", spaceToDepthOp->getName(), spaceToDepthOp->getLoc());

    // insert copy before SpaceToDepth
    auto memRefInputType = spaceToDepthOp.input().getType().cast<mlir::MemRefType>();
    auto cmxIndexSymbolAttr = IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
    auto newMemRefInputType = mlir::MemRefType::get(memRefInputType.getShape(), memRefInputType.getElementType(),
                                                    memRefInputType.getLayout(), cmxIndexSymbolAttr);
    auto allocInputOp = rewriter.create<mlir::memref::AllocOp>(spaceToDepthOp->getLoc(), newMemRefInputType);
    auto inputCopyOp = rewriter.create<VPUIP::CopyOp>(spaceToDepthOp->getLoc(), spaceToDepthOp.input(), allocInputOp);
    _log.trace("Insert copy Op before SpaceToDepthDMA Op with alloc buffer location {0}.",
               newMemRefInputType.getMemorySpace());

    // create new SpaceToDepthDMA Op
    auto spaceToDepthMemRefType = spaceToDepthOp.getType();
    auto newSpaceToDepthMemRefType =
            mlir::MemRefType::get(spaceToDepthMemRefType.getShape(), spaceToDepthMemRefType.getElementType(),
                                  spaceToDepthMemRefType.getLayout(), cmxIndexSymbolAttr);
    auto allocSpaceToDepthOp =
            rewriter.create<mlir::memref::AllocOp>(spaceToDepthOp->getLoc(), newSpaceToDepthMemRefType);
    auto newSpaceToDepthOp = rewriter.create<VPUIP::SpaceToDepthDMAOp>(
            spaceToDepthOp->getLoc(), inputCopyOp.output(), allocSpaceToDepthOp, spaceToDepthOp.block_sizeAttr(),
            spaceToDepthOp.modeAttr(), nullptr);
    _log.trace("Create new SpaceToDepthDMA Op with alloc buffer location {0}.",
               newSpaceToDepthMemRefType.getMemorySpace());

    // create copy after SpaceToDepth
    auto memRefOutputType = spaceToDepthOp.output().getType().cast<mlir::MemRefType>();
    auto newMemRefOuputType =
            mlir::MemRefType::get(memRefOutputType.getShape(), memRefOutputType.getElementType(),
                                  memRefOutputType.getLayout(), IndexedSymbolAttr::get(rewriter.getContext(), "DDR"));
    auto allocOutputOp = rewriter.create<mlir::memref::AllocOp>(spaceToDepthOp->getLoc(), newMemRefOuputType);
    rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(spaceToDepthOp, newSpaceToDepthOp->getResult(0), allocOutputOp);
    _log.trace("Insert copy Op after SpaceToDepthDMA Op with alloc buffer location {0}.",
               newMemRefOuputType.getMemorySpace());

    return mlir::success();
}

//
// PerAxisTileConverter
//

class ConvertToDMAPass::PerAxisTileConverter final : public mlir::OpRewritePattern<VPUIP::PerAxisTileUPAOp> {
public:
    PerAxisTileConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::PerAxisTileUPAOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::PerAxisTileConverter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::PerAxisTileUPAOp perAxisTileOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::PerAxisTileConverter::matchAndRewrite(VPUIP::PerAxisTileUPAOp perAxisTileOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("PerAxisTileOp rewriter operation '{0}' at '{1}'", perAxisTileOp->getName(), perAxisTileOp->getLoc());

    // insert copy before PerAxisTileDMA
    auto memRefInputType = perAxisTileOp.input().getType().cast<mlir::MemRefType>();
    auto cmxIndexSymbolAttr = IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
    auto newMemRefInputType = mlir::MemRefType::get(memRefInputType.getShape(), memRefInputType.getElementType(),
                                                    memRefInputType.getLayout(), cmxIndexSymbolAttr);
    auto allocInputOp = rewriter.create<mlir::memref::AllocOp>(perAxisTileOp->getLoc(), newMemRefInputType);
    auto inputCopyOp = rewriter.create<VPUIP::CopyOp>(perAxisTileOp->getLoc(), perAxisTileOp.input(), allocInputOp);
    _log.trace("Insert copy Op before PerAxisTileDMA Op with alloc buffer location {0}.",
               newMemRefInputType.getMemorySpace());

    // create new PerAxisTileDMA Op
    auto perAxisTileMemRefType = perAxisTileOp.getType();
    auto newPerAxisTileMemRefType =
            mlir::MemRefType::get(perAxisTileMemRefType.getShape(), perAxisTileMemRefType.getElementType(),
                                  perAxisTileMemRefType.getLayout(), cmxIndexSymbolAttr);
    auto allocPerAxisTileOp = rewriter.create<mlir::memref::AllocOp>(perAxisTileOp->getLoc(), newPerAxisTileMemRefType);
    auto newperAxisTileOp =
            rewriter.create<VPUIP::PerAxisTileDMAOp>(perAxisTileOp->getLoc(), inputCopyOp.output(), allocPerAxisTileOp,
                                                     perAxisTileOp.axisAttr(), perAxisTileOp.tilesAttr(), nullptr);
    _log.trace("Create new PerAxisTileDMA Op with alloc buffer location {0}.",
               newPerAxisTileMemRefType.getMemorySpace());

    // create copy after perAxisTile
    auto memRefOutputType = perAxisTileOp.output().getType().cast<mlir::MemRefType>();
    auto newMemRefOuputType =
            mlir::MemRefType::get(memRefOutputType.getShape(), memRefOutputType.getElementType(),
                                  memRefOutputType.getLayout(), IndexedSymbolAttr::get(rewriter.getContext(), "DDR"));
    auto allocOutputOp = rewriter.create<mlir::memref::AllocOp>(perAxisTileOp->getLoc(), newMemRefOuputType);
    rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(perAxisTileOp, newperAxisTileOp->getResult(0), allocOutputOp);
    _log.trace("Insert copy Op after PerAxisTileDMA Op with alloc buffer location {0}.",
               newMemRefOuputType.getMemorySpace());

    return mlir::success();
}

//
//  SwKernelMemPermuteConverter
//

class ConvertToDMAPass::SwKernelMemPermuteConverter final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    SwKernelMemPermuteConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::SwKernelMemPermOp");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::SwKernelMemPermuteConverter::matchAndRewrite(
        VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const {
    if (!VPUIP::isMemPermSwKernel(swKernelOp)) {
        return mlir::failure();
    }

    _log.trace("Got Mempermute SwKernel '{0}' at '{1}'", swKernelOp->getName(), swKernelOp->getLoc());

    auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp);
    VPUX_THROW_UNLESS(memPerm.has_value(), "Cannot extract mem_perm attribute from permute SwKernel '{0}'.",
                      swKernelOp.getLoc());

    VPUX_THROW_UNLESS(swKernelOp->getNumOperands() == 2, "Unexpected operand number for VPUIP.SwKernelOp at '{0}'",
                      swKernelOp);

    const auto inType = swKernelOp.getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto input = swKernelOp.getOperand(0);
    const auto outputBuf = swKernelOp.getOperand(1);
    // Check for inversed permutation which needs split into 2 consecutive permuteDMAs
    // e.g. pattern [d0, d1, d2, d3] -> [d0, d3, d2, d1]
    if (!VPUIP::isSplitNeededForPermuteDMA(inType, memPerm.value())) {
        rewriter.replaceOpWithNewOp<VPUIP::PermuteDMAOp>(swKernelOp, input, outputBuf,
                                                         mlir::AffineMapAttr::get(memPerm.value()), nullptr);

        _log.nest().trace("Rewrite Mempermute SwKernel '{0}' at '{1}' to PermuteDMA.", swKernelOp->getName(),
                          swKernelOp->getLoc());
        return mlir::success();
    }
    _log.nest().trace("Split into 2 permuteDMA: memPerm {0}", memPerm);

    auto permuteMemRefType = swKernelOp.getOperand(1).getType().dyn_cast<mlir::MemRefType>();
    VPUX_THROW_WHEN(permuteMemRefType == nullptr, "Unexpected output type for VPUIP.SwKernelOp at '{0}'",
                    swKernelOp->getLoc());

    // 3 types of memPermute can be supported:
    // a) NHWC -> NCHW; b) NWCH -> NCHW; c) NCWH -> NHWC
    // The first 2 types cannot be replaced with the 2 consecutive permuteDMAs directly,
    // so a permuteCast is required before the 2 permuteDMAs.
    // The dst order of permuteCast can be reversedly derived from the final dst order

    const auto outType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto outShapedType = outType.cast<vpux::NDTypeInterface>();
    auto outOrder = outType.getDimsOrder();
    // The 2nd permuteDMA is [d0, d2, d3, d1] -> [d0, d3, d2, d1], permutation is [d0, d2, d1, d3]
    // The inversed permutation is [d0, d2, d1, d3]
    // dstOutOrder here is actually the inOrder for the second permuation
    auto inversePermLast = mlir::AffineMap::getPermutationMap({0, 2, 1, 3}, rewriter.getContext());
    auto dstOutOrder = applyPermutation(outOrder, DimsOrder::fromAffineMap(inversePermLast));
    // The 1st permuteDMA is [d0, d1, d2, d3] -> [d0, d2, d3, d1], permutation is [d0, d2, d3, d1]
    // The inverse permuation is [d0, d3, d1, d2]]
    auto inversePermFirst = mlir::AffineMap::getPermutationMap({0, 3, 1, 2}, rewriter.getContext());
    auto permuteCastDstOrder = applyPermutation(dstOutOrder, DimsOrder::fromAffineMap(inversePermFirst));
    auto permuteCastOutType = outShapedType.changeDimsOrder(permuteCastDstOrder);
    _log.nest().trace("Deduced permuteCastDstOrder = {0} from outOrder {1}", permuteCastDstOrder, outOrder);
    auto permuteCastOp = rewriter.create<VPUIP::PermuteCastOp>(
            swKernelOp->getLoc(), permuteCastOutType, input,
            mlir::AffineMapAttr::get(permuteCastDstOrder.toAffineMap(rewriter.getContext())),
            mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(rewriter.getContext())));

    // create the 1st permuteDMA Op [d0, d1, d2, d3] -> [d0, d2, d3, d1], permutation is [d0, d2, d3, d1]
    auto memPermFirst = mlir::AffineMap::getPermutationMap({0, 2, 3, 1}, rewriter.getContext());
    auto newPermuteMemRefType = mlir::MemRefType::get(
            permuteMemRefType.getShape(), permuteMemRefType.getElementType(),
            dstOutOrder.toAffineMap(rewriter.getContext()),
            IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0));

    auto allocPermuteOp = rewriter.create<mlir::memref::AllocOp>(swKernelOp->getLoc(), newPermuteMemRefType);
    auto permuteDmaOp = rewriter.create<VPUIP::PermuteDMAOp>(swKernelOp->getLoc(), permuteCastOp, allocPermuteOp,
                                                             mlir::AffineMapAttr::get(memPermFirst), nullptr);

    // create the 2nd permuteDMA Op [d0, d2, d3, d1] -> [d0, d3, d2, d1], permutation is [d0, d2, d1, d3]
    auto memPermLast = mlir::AffineMap::getPermutationMap({0, 2, 1, 3}, rewriter.getContext());
    rewriter.replaceOpWithNewOp<VPUIP::PermuteDMAOp>(swKernelOp, permuteDmaOp.output(), outputBuf,
                                                     mlir::AffineMapAttr::get(memPermLast), nullptr);
    _log.nest().trace("Rewrite Mempermute SwKernel '{0}' at '{1}' to 2 PermuteDMA ops.", swKernelOp->getName(),
                      swKernelOp->getLoc());
    return mlir::success();
}

//
// SwKernelDepthToSpaceConverter
//

class ConvertToDMAPass::SwKernelDepthToSpaceConverter final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    SwKernelDepthToSpaceConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::SwKernelDepthToSpaceOp");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::SwKernelDepthToSpaceConverter::matchAndRewrite(
        VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const {
    if (!VPUIP::isDepthToSpaceSwKernel(swKernelOp)) {
        return mlir::failure();
    }

    _log.trace("Got DepthToSpace SwKernel '{0}' at '{1}'", swKernelOp->getName(), swKernelOp->getLoc());

    auto depthToSpaceAttrs = VPUIP::getDepthToSpaceSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(depthToSpaceAttrs.has_value(),
                      "Cannot extract depthToSpace attribute from depthToSpace SwKernel '{0}'.", swKernelOp.getLoc());
    auto modeAttr = std::get<0>(depthToSpaceAttrs.value());
    auto blockSizeAttr = std::get<1>(depthToSpaceAttrs.value());
    auto paddedChannel = std::get<2>(depthToSpaceAttrs.value());

    VPUX_THROW_UNLESS(swKernelOp->getNumOperands() == 2, "Unexpected operand number for VPUIP.SwKernelOp at '{0}'",
                      swKernelOp);
    auto input = swKernelOp.getOperand(0);
    auto outputBuf = swKernelOp.getOperand(1);

    rewriter.replaceOpWithNewOp<VPUIP::DepthToSpaceDMAOp>(swKernelOp, input, outputBuf, blockSizeAttr, modeAttr,
                                                          nullptr, paddedChannel);

    _log.nest().trace("Rewrite DepthToSpace SwKernel '{0}' at '{1}' to DepthToSpaceDMA.", swKernelOp->getName(),
                      swKernelOp->getLoc());
    return mlir::success();
}

//
// SwKernelSpaceToDepthConverter
//

class ConvertToDMAPass::SwKernelSpaceToDepthConverter final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    SwKernelSpaceToDepthConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::SwKernelSpaceToDepthOp");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::SwKernelSpaceToDepthConverter::matchAndRewrite(
        VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const {
    if (!VPUIP::isSpaceToDepthSwKernel(swKernelOp)) {
        return mlir::failure();
    }

    _log.trace("Got SpaceToDepth SwKernel '{0}' at '{1}'", swKernelOp->getName(), swKernelOp->getLoc());

    auto spaceToDepthAttrs = VPUIP::getSpaceToDepthSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(spaceToDepthAttrs.has_value(),
                      "Cannot extract spaceToDepth attribute from spaceToDepth SwKernel '{0}'.", swKernelOp.getLoc());
    auto modeAttr = spaceToDepthAttrs.value().first;
    auto blockSizeAttr = spaceToDepthAttrs.value().second;

    VPUX_THROW_UNLESS(swKernelOp->getNumOperands() == 2, "Unexpected operand number for VPUIP.SwKernelOp at '{0}'",
                      swKernelOp);
    auto input = swKernelOp.getOperand(0);
    auto outputBuf = swKernelOp.getOperand(1);

    rewriter.replaceOpWithNewOp<VPUIP::SpaceToDepthDMAOp>(swKernelOp, input, outputBuf, blockSizeAttr, modeAttr,
                                                          nullptr);

    _log.nest().trace("Rewrite SpaceToDepth SwKernel '{0}' at '{1}' to SpaceToDepthDMA.", swKernelOp->getName(),
                      swKernelOp->getLoc());
    return mlir::success();
}

//
// ExpandConverter
//

class ConvertToDMAPass::ExpandConverter final : public mlir::OpRewritePattern<VPUIP::ExpandOp> {
public:
    ExpandConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::ExpandOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::ExpandConverter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandOp depthToSpaceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::ExpandConverter::matchAndRewrite(VPUIP::ExpandOp expandOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Expand '{0}' at '{1}'", expandOp->getName(), expandOp->getLoc());

    auto inputType = expandOp.input().getType().cast<NDTypeInterface>();
    VPUX_THROW_WHEN(inputType.getElementType().isa<mlir::FloatType>(),
                    "Only Non Float type ExpandOp can be converted to DMA, but got '{0}'", inputType.getElementType());

    const auto outputType = expandOp.output().getType().cast<vpux::NDTypeInterface>();
    _log.nest().trace("inType: '{0}', outType: '{1}', padBegin: '{2}', padEnd: '{3}'", inputType, outputType,
                      expandOp.pads_begin(), expandOp.pads_end());

    auto newMemRefOutputType = outputType;
    auto expandedBuffer =
            rewriter.create<mlir::memref::AllocOp>(expandOp.getLoc(), newMemRefOutputType.cast<mlir::MemRefType>());

    rewriter.replaceOpWithNewOp<VPUIP::ExpandDMAOp>(expandOp, expandOp.input(), expandedBuffer,
                                                    expandOp.pads_beginAttr(), expandOp.pads_endAttr(), nullptr);

    _log.nest().trace("Rewrite Expand '{0}' at '{1}' to ExpandDMAOp.", expandOp->getName(), expandOp->getLoc());

    return mlir::success();
}

//
// SwKernelPerAxisTileConverter
//

class ConvertToDMAPass::SwKernelPerAxisTileConverter final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    SwKernelPerAxisTileConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::SwKernelPerAxisTileOp");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::SwKernelPerAxisTileConverter::matchAndRewrite(
        VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const {
    if (!VPUIP::isTileSwKernel(swKernelOp)) {
        return mlir::failure();
    }

    _log.trace("Got Tile SwKernel '{0}' at '{1}'", swKernelOp->getName(), swKernelOp->getLoc());
    const auto ctx = swKernelOp->getContext();

    const auto inType = swKernelOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto outType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(inType.getRank() == outType.getRank(), "Tile Op has different input '{0}' output '{1}' rank",
                      inType, outType);

    auto lastResult = swKernelOp->getOperand(0);
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();
    // If Tile Op repeats at more than one Axis, will convert to PerAxisTile
    // e.g. Input 1x2x3x4, Output 1x4x9x16, repeats [1x2x3x4]
    // Convert to 3 sub PerAxisTileDMA Op:
    // Sub Op 0: Input 1x2x3x4, Output 1x4x3x4,  repeats [1x2x1x1]
    // Sub Op 1: Input 1x4x3x4, Output 1x4x9x4,  repeats [1x1x3x1]
    // Sub Op 2: Input 1x4x9x4, Output 1x4x9x16, repeats [1x1x1x4]
    for (size_t idx = 0; idx < checked_cast<size_t>(inType.getRank()); ++idx) {
        if (inShape[Dim(idx)] == outShape[Dim(idx)]) {
            continue;
        }

        auto lastInType = lastResult.getType().cast<vpux::NDTypeInterface>();
        auto newOutShape = to_small_vector(lastInType.getShape());
        newOutShape[idx] = outShape[Dim(idx)];
        auto newMemRefOutputType = outType.changeShape(ShapeRef(newOutShape));
        auto outputBuffer = rewriter.create<mlir::memref::AllocOp>(swKernelOp->getLoc(),
                                                                   newMemRefOutputType.cast<mlir::MemRefType>());

        VPUX_THROW_UNLESS(outShape[Dim(idx)] % inShape[Dim(idx)] == 0 && outShape[Dim(idx)] / inShape[Dim(idx)] > 1,
                          "Unexpect Tile Op inshape '{0}' outShape '{1}' rank", inShape, outShape);
        const auto repeats = outShape[Dim(idx)] / inShape[Dim(idx)];
        const auto repeatsAttr = mlir::IntegerAttr::get(getInt64Type(ctx), repeats);
        const auto axisAttr = mlir::IntegerAttr::get(getInt64Type(ctx), idx);

        lastResult = rewriter.create<VPUIP::PerAxisTileDMAOp>(swKernelOp->getLoc(), lastResult, outputBuffer, axisAttr,
                                                              repeatsAttr, nullptr)
                             .getResult();
    }

    VPUX_THROW_UNLESS(lastResult != swKernelOp->getOperand(0), "Unexpect Tile Op at '{0}', not find repeats Axis",
                      swKernelOp->getLoc());
    rewriter.replaceOp(swKernelOp, lastResult);

    _log.nest().trace("Rewrite PerAxisTile SwKernel '{0}' at '{1}' to PerAxisTileDMA.", swKernelOp->getName(),
                      swKernelOp->getLoc());
    return mlir::success();
}

//
// UpsamplingOpConverter
//

class ConvertToDMAPass::UpsamplingOpConverter final : public mlir::OpRewritePattern<VPUIP::UpsamplingUPAOp> {
public:
    UpsamplingOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::UpsamplingUPAOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::UpsamplingUPAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::UpsamplingOpConverter::matchAndRewrite(VPUIP::UpsamplingUPAOp origOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Upsampling Op {0}", origOp->getLoc());

    const auto outputShape = getShape(origOp.output());

    const auto upsamplingFactorVectorTmp = parseIntArrayAttr<int64_t>(origOp.upsampling_factor());
    SmallVector<int64_t> upsamplingFactorVector = {1, upsamplingFactorVectorTmp[2], upsamplingFactorVectorTmp[1],
                                                   upsamplingFactorVectorTmp[0]};

    auto constZeros = VPU::getZerosConst(rewriter, outputShape.toValues(), origOp.input(), origOp.getLoc());

    auto copyZeroOp = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), constZeros, origOp.output_buff());
    auto upsampleFactorAttr = getIntArrayAttr(origOp.getContext(), upsamplingFactorVector);
    auto upsampeDMA = rewriter.create<VPUIP::UpsamplingDMAOp>(origOp.getLoc(), origOp.input(), copyZeroOp.output(),
                                                              upsampleFactorAttr, nullptr, nullptr,
                                                              getIntAttr(origOp->getContext(), 0), nullptr, nullptr);

    rewriter.replaceOp(origOp, upsampeDMA.output());
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertToDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (!mlir::isa<VPUIP::DepthToSpaceUPAOp, VPUIP::SpaceToDepthUPAOp, VPUIP::PermuteUPAOp, VPUIP::PerAxisTileUPAOp,
                       VPUIP::UpsamplingUPAOp, VPUIP::SwKernelOp>(op)) {
            return true;
        }

        if (!VPUIP::isLegalAndBeneficialConvertToDMA(op, _log)) {
            return true;
        }

        if (mlir::isa<VPUIP::SwKernelOp>(op)) {
            return false;
        }

        const auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();

        return !(inputType.getMemoryKind() == VPU::MemoryKind::DDR &&
                 outputType.getMemoryKind() == VPU::MemoryKind::DDR);
    });
    target.addIllegalOp<VPUIP::ExpandOp>();
    target.addLegalOp<VPUIP::ExpandDMAOp>();

    target.addLegalOp<mlir::memref::AllocOp>();
    target.addLegalOp<VPUIP::CopyOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<VPUIP::UpsamplingDMAOp>();
    target.addIllegalOp<VPUIP::UpsamplingUPAOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DepthToSpaceConverter>(&ctx, _log);
    patterns.add<MemPermuteConverter>(&ctx, _log);
    patterns.add<SpaceToDepthConverter>(&ctx, _log);
    patterns.add<ExpandConverter>(&ctx, _log);
    patterns.add<PerAxisTileConverter>(&ctx, _log);
    patterns.add<SwKernelMemPermuteConverter>(&ctx, _log);
    patterns.add<SwKernelDepthToSpaceConverter>(&ctx, _log);
    patterns.add<SwKernelSpaceToDepthConverter>(&ctx, _log);
    patterns.add<SwKernelPerAxisTileConverter>(&ctx, _log);
    patterns.add<UpsamplingOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToDMACMXPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertToDMAPass(Logger log) {
    return std::make_unique<ConvertToDMAPass>(log);
}

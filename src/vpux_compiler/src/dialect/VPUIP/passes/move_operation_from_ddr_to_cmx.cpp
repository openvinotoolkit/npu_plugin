//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/permute_as_nndma_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;
namespace {

//
// MoveOperationFromDDRtoCMXPass
//

class MoveOperationFromDDRtoCMXPass final : public VPUIP::MoveOperationFromDDRtoCMXBase<MoveOperationFromDDRtoCMXPass> {
public:
    explicit MoveOperationFromDDRtoCMXPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void getDependentDialects(mlir::DialectRegistry& registry) const override {
        registry.insert<vpux::VPUIP::VPUIPDialect>();
    }

public:
    class DepthToSpaceConverter;
    class MemPermuteConverter;

private:
    void safeRunOnFunc() final;
};

//
// DepthToSpaceConverter
//

class MoveOperationFromDDRtoCMXPass::DepthToSpaceConverter final :
        public mlir::OpRewritePattern<VPUIP::DepthToSpaceUPAOp> {
public:
    DepthToSpaceConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::DepthToSpaceUPAOp>(ctx), _log(log) {
        setDebugName("MoveOperationFromDDRtoCMXPass::DepthToSpaceConverter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::DepthToSpaceUPAOp depthToSpaceOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveOperationFromDDRtoCMXPass::DepthToSpaceConverter::matchAndRewrite(
        VPUIP::DepthToSpaceUPAOp depthToSpaceOp, mlir::PatternRewriter& rewriter) const {
    const auto inputType = depthToSpaceOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = depthToSpaceOp.output().getType().cast<vpux::NDTypeInterface>();

    // This pass will move DepthToSpace from DDR to CMX
    // The total memory should be fitted into CMX
    Byte requiredCMX(0);
    requiredCMX += inputType.getTotalAllocSize();
    requiredCMX += outputType.getTotalAllocSize();
    VPUX_THROW_UNLESS(requiredCMX < VPU::getTotalCMXSize(depthToSpaceOp.getOperation()),
                      "Memeory size of depthToSpaceOp is larger than CMX");

    const auto outputChannelsAttr = getIntAttr(depthToSpaceOp.getContext(), outputType.getShape()[Dims4D::Act::C]);
    const auto outputWidthAttr = getIntAttr(depthToSpaceOp.getContext(), outputType.getShape()[Dims4D::Act::W]);

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
            depthToSpaceOp.modeAttr(), outputChannelsAttr, outputWidthAttr);
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

class MoveOperationFromDDRtoCMXPass::MemPermuteConverter final : public mlir::OpRewritePattern<VPUIP::PermuteUPAOp> {
public:
    MemPermuteConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::PermuteUPAOp>(ctx), _log(log) {
        setDebugName("MoveOperationFromDDRtoCMXPass::MemPermuteConverter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::PermuteUPAOp permuteUPAOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveOperationFromDDRtoCMXPass::MemPermuteConverter::matchAndRewrite(
        VPUIP::PermuteUPAOp permuteUPAOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("MemPermute rewriter operation '{0}' at '{1}'", permuteUPAOp->getName(), permuteUPAOp->getLoc());

    // Check memory size and location
    const auto outputType = permuteUPAOp.output().getType().cast<vpux::NDTypeInterface>();
    if (outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
        _log.trace("MemPermute Op {0} is already in CMX.", permuteUPAOp);
        rewriter.replaceOpWithNewOp<VPUIP::PermuteDMAOp>(permuteUPAOp, permuteUPAOp.input(), permuteUPAOp.output_buff(),
                                                         getIntAttr(permuteUPAOp.getContext(), 0),
                                                         permuteUPAOp.order_valueAttr());
        return mlir::success();
    }

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
    auto permuteMemRefType = permuteUPAOp.getType();
    auto newPermuteMemRefType = mlir::MemRefType::get(
            permuteMemRefType.getShape(), permuteMemRefType.getElementType(), permuteMemRefType.getLayout(),
            IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0));
    auto allocPermuteOp = rewriter.create<mlir::memref::AllocOp>(permuteUPAOp->getLoc(), newPermuteMemRefType);
    auto newMemPermuteOp = rewriter.create<VPUIP::PermuteDMAOp>(
            permuteUPAOp->getLoc(), inputCopyOp.output(), allocPermuteOp, getIntAttr(permuteUPAOp.getContext(), 0),
            permuteUPAOp.order_valueAttr());
    _log.trace("Create new PermuteDMA Op with alloc buffer location {0}.", newPermuteMemRefType.getMemorySpace());

    // create copy after MemPermuteDMAOp
    auto memRefOutputType = permuteUPAOp.output().getType().cast<mlir::MemRefType>();
    auto newMemRefOuputType =
            mlir::MemRefType::get(memRefOutputType.getShape(), memRefOutputType.getElementType(),
                                  memRefOutputType.getLayout(), IndexedSymbolAttr::get(rewriter.getContext(), "DDR"));
    auto allocOutputOp = rewriter.create<mlir::memref::AllocOp>(permuteUPAOp->getLoc(), newMemRefOuputType);
    rewriter.replaceOpWithNewOp<VPUIP::CopyOp>(permuteUPAOp, newMemPermuteOp->getResult(0), allocOutputOp);
    _log.trace("Insert copy Op after MemPermute Op with alloc buffer location {0}.",
               newMemRefOuputType.getMemorySpace());

    return mlir::success();
}

//
// safeRunOnFunc
//

void MoveOperationFromDDRtoCMXPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto module = getOperation();
    const auto arch = VPU::getArch(module);
    if (arch != VPU::ArchKind::VPUX30XX) {
        _log.trace("MoveOperationFromDDRtoCMXPass enabled only for VPUX30XX device. Got: {0}", arch);
        return;
    }

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto depthToSpaceOp = mlir::dyn_cast<VPUIP::DepthToSpaceUPAOp>(op)) {
            _log.trace("Got DepthToSpaceOp Op {0}.", depthToSpaceOp);
            const auto inputType = depthToSpaceOp.input().getType().cast<vpux::NDTypeInterface>();
            const auto outputType = depthToSpaceOp.output().getType().cast<vpux::NDTypeInterface>();

            // At present, depthToSpaceOp can support layout with NCHW and NHWC in VPUX compiler
            // This is ticket E#41656 to support NCHW layout optimization
            auto inOrder = inputType.getDimsOrder();
            auto outOrder = outputType.getDimsOrder();
            if (inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NHWC) {
                return false;
            }

            _log.trace("Can't convert DepthToSpaceOp inOrder {0}, outOrder {1} to DepthToSpaceDMA.", inOrder, outOrder);
        } else if (auto permuteUPAOp = mlir::dyn_cast<VPUIP::PermuteUPAOp>(op)) {
            if (!VPUIP::getPermuteDMASubShapes(permuteUPAOp, _log).hasValue()) {
                _log.trace("MemPermute Op {0} doesn't support DMA implementation.", permuteUPAOp);
                return true;
            }
            if (!VPUIP::isBeneficialForUsingDMA(permuteUPAOp, _log)) {
                _log.trace("MemPermute Op {0} isn't beneficial for using DMA.", permuteUPAOp);
                return true;
            }

            // Check memory size can fit in CMX or not
            const auto inputType = permuteUPAOp.input().getType().cast<vpux::NDTypeInterface>();
            const auto outputType = permuteUPAOp.output().getType().cast<vpux::NDTypeInterface>();
            if (outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
                return false;
            }
            Byte requiredCMX(0);
            requiredCMX += inputType.getTotalAllocSize();
            requiredCMX += outputType.getTotalAllocSize();
            if (requiredCMX > VPU::getTotalCMXSize(permuteUPAOp.getOperation())) {
                _log.trace("Memory size of MemPermute Op {0} is larger than CMX, can not move to CMX.", permuteUPAOp);
                return true;
            }
            return false;
        }
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<DepthToSpaceConverter>(&ctx, _log);
    patterns.insert<MemPermuteConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveOperationFromDDRtoCMXPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createMoveOperationFromDDRtoCMXPass(Logger log) {
    return std::make_unique<MoveOperationFromDDRtoCMXPass>(log);
}

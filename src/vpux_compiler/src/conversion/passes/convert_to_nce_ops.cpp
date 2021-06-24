//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/enums.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// Utilities
//

const EnumMap<VPUIP::ArchKind, VPUIP::MPEMode> mpeMap = {
        {VPUIP::ArchKind::VPU3400_A0, VPUIP::MPEMode::VECTOR_FP16},  //
        {VPUIP::ArchKind::VPU3400, VPUIP::MPEMode::VECTOR_FP16},     //
        {VPUIP::ArchKind::VPU3700, VPUIP::MPEMode::VECTOR_FP16},     //
        {VPUIP::ArchKind::VPU3900, VPUIP::MPEMode::VECTOR_FP16},     //
        {VPUIP::ArchKind::VPU3720, VPUIP::MPEMode::CUBOID_16x16},    //
};

mlir::Value createWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, int64_t OC, mlir::Value weights,
                                     mlir::Value bias, mlir::Value activationWindow) {
    SmallVector<int64_t> weightTableShape{OC, 1, 1, VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};

    auto* ctx = builder.getContext();

    const auto dataType = mlir::MemRefType::get(weightTableShape, getSInt32Type(builder.getContext()));
    auto createWeightsTableOp = builder.create<VPUIP::WeightsTableOp>(loc, dataType, weights, bias, activationWindow);

    const auto cmxMemSpaceAttr = VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN);
    const auto dataTypeCMX = changeMemSpace(dataType, cmxMemSpaceAttr);

    auto dataAllocOp = builder.create<mlir::memref::AllocOp>(loc, dataTypeCMX);
    auto copyOp = builder.create<IERT::CopyOp>(loc, createWeightsTableOp.output(), dataAllocOp);

    return copyOp.output();
}

mlir::Value prepareInputForDPU(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input) {
    // DMA DDR -> CMX
    const auto origType = input.getType().cast<mlir::MemRefType>();
    auto typeCMX = changeMemSpace(origType,
                                  VPUIP::PhysicalMemoryAttr::get(builder.getContext(), VPUIP::PhysicalMemory::CMX_NN));
    auto dmaAllocOp = builder.create<mlir::memref::AllocOp>(loc, typeCMX);
    auto dmaOp = builder.create<IERT::CopyOp>(loc, input, dmaAllocOp.memref());

    return dmaOp.output();
}

void addDPUTasks(VPUIP::NCEClusterTaskOp nceOp, mlir::PatternRewriter& rewriter, int32_t numDPU,
                 ArrayRef<int64_t> opPadsBegin, ArrayRef<int64_t> opPadsEnd, VPUIP::MPEMode mpeMode) {
    auto* ctx = nceOp.getContext();

    const auto outputShape = getShape(nceOp.output());
    const auto dpuTiles = VPUIP::DpuTiler::tileOverH(numDPU, outputShape, opPadsBegin, opPadsEnd);

    for (const auto& dpuTile : dpuTiles) {
        const auto startAttr = getInt32ArrayAttr(ctx, makeArrayRef(dpuTile.start));
        const auto endAttr = getInt32ArrayAttr(ctx, makeArrayRef(dpuTile.end));

        const auto padsBeginAttr = getInt32ArrayAttr(ctx, dpuTile.padsBegin);
        const auto padsEndAttr = getInt32ArrayAttr(ctx, dpuTile.padsEnd);

        nceOp.addDPUTask(rewriter, startAttr, endAttr, padsBeginAttr, padsEndAttr, mpeMode);
    }
}

//
// ConvRewrite
//

class ConvRewrite final : public mlir::OpRewritePattern<IERT::ConvolutionOp> {
public:
    ConvRewrite(mlir::MLIRContext* ctx, uint32_t numDPU, vpux::VPUIP::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IERT::ConvolutionOp>(ctx), _numDPU(numDPU), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const uint32_t _numDPU;
    vpux::VPUIP::ArchKind _arch;
    Logger _log;
};

mlir::Value prepareFilterForDPU(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value filter) {
    const auto origType = filter.getType().cast<mlir::MemRefType>();

    // DMA DDR -> CMX
    const auto typeCMX = changeMemSpace(
            origType, VPUIP::PhysicalMemoryAttr::get(builder.getContext(), VPUIP::PhysicalMemory::CMX_NN));
    auto dmaAllocOp = builder.create<mlir::memref::AllocOp>(loc, typeCMX);
    auto dmaOp = builder.create<IERT::CopyOp>(loc, filter, dmaAllocOp.memref());

    return dmaOp.output();
}

mlir::LogicalResult ConvRewrite::matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    if (VPUIP::NCEInvariant::verifyOp(origOp, _log).failed()) {
        return matchFailed(rewriter, origOp, "Operation {0} does not satisfy the NCE invariant", origOp);
    }

    //
    // Get dimensions
    //

    const auto filterShape = getShape(origOp.filter());

    const auto OC = filterShape[IERT::ConvolutionOp::filter_out_channel_dim()];
    const auto KY = filterShape[IERT::ConvolutionOp::filter_spatial_height_dim()];
    const auto KX = filterShape[IERT::ConvolutionOp::filter_spatial_width_dim()];

    //
    // Prepare input for DPU
    //

    auto inputDPU = prepareInputForDPU(rewriter, origOp->getLoc(), origOp.input());
    auto filterDPU = prepareFilterForDPU(rewriter, origOp->getLoc(), origOp.filter());
    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), OC, filterDPU, origOp.bias(), nullptr);

    //
    // Prepare output buffer for DPU
    //

    const auto origOutType = origOp.output().getType().cast<mlir::MemRefType>();
    const auto outReorderType = changeDimsOrder(origOutType, DimsOrder::NHWC);
    const auto outTypeCMX =
            changeMemSpace(outReorderType, VPUIP::PhysicalMemoryAttr::get(getContext(), VPUIP::PhysicalMemory::CMX_NN));

    auto outAllocOpCMX = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), outTypeCMX);

    //
    // Create NCE per-cluster Operation
    //

    const auto padsBegin = parseIntArrayAttr(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr(origOp.pads_end());
    const auto kernelPaddingAttr =
            getInt32ArrayAttr(getContext(), makeArrayRef({padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]}));

    const auto kernelSizeAttr = getInt32ArrayAttr(getContext(), makeArrayRef({KX, KY}));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), inputDPU, filterDPU, weightsTable, /*activation_window=*/nullptr,
            /*parent_input=*/inputDPU,
            /*parent_output=*/outAllocOpCMX.memref(),
            /*output_buff=*/outAllocOpCMX.memref(), VPUIP::NCETaskType::CONV, kernelSizeAttr, origOp.strides(),
            kernelPaddingAttr, /*activation_window_channel_length=*/nullptr);

    //
    // Create DPU sub-task
    //

    addDPUTasks(nceOp, rewriter, _numDPU, padsBegin, padsEnd, mpeMap.at(_arch));

    //
    // DMA output CMX -> DDR
    //

    rewriter.replaceOpWithNewOp<IERT::CopyOp>(origOp, nceOp.output(), origOp.output_buff());

    return mlir::success();
}

//
// MaxPoolRewrite
//

class MaxPoolRewrite final : public mlir::OpRewritePattern<IERT::MaxPoolOp> {
public:
    MaxPoolRewrite(mlir::MLIRContext* ctx, uint32_t numDPU, vpux::VPUIP::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IERT::MaxPoolOp>(ctx), _numDPU(numDPU), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const uint32_t _numDPU;
    vpux::VPUIP::ArchKind _arch;
    Logger _log;
};

mlir::Value createActivationWindowTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<uint8_t> fakeSparsity,
                                         int64_t numChannels) {
    auto* ctx = builder.getContext();
    const auto elemType = getUInt8Type(builder.getContext());

    SmallVector<int64_t> fakeSparsityShape{numChannels, 1, 1, static_cast<int64_t>(fakeSparsity.size()) / numChannels};

    const auto dataStorageType = mlir::RankedTensorType::get(fakeSparsityShape, elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, fakeSparsity);

    const auto dataType = mlir::MemRefType::get(fakeSparsityShape, elemType);
    auto dataConstOp = builder.create<Const::DeclareOp>(loc, dataType, Const::ContentAttr::get(dataAttr));

    const auto cmxMemSpaceAttr = VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN);
    const auto dataTypeCMX = changeMemSpace(dataType, cmxMemSpaceAttr);

    auto dataAllocOp = builder.create<mlir::memref::AllocOp>(loc, dataTypeCMX);
    auto copyOp = builder.create<IERT::CopyOp>(loc, dataConstOp.output(), dataAllocOp);

    return copyOp.output();
}

mlir::LogicalResult MaxPoolRewrite::matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    if (VPUIP::NCEInvariant::verifyOp(origOp, _log).failed()) {
        return matchFailed(rewriter, origOp, "Operation {0} does not satisfy the NCE invariant", origOp);
    }

    //
    // Get dimensions
    //

    const auto origInputType = origOp.input().getType().cast<mlir::MemRefType>();
    const auto inputShape = getShape(origInputType);

    const auto IC = inputShape[IERT::MaxPoolOp::act_channel_dim()];

    const auto kernelSize = parseIntArrayAttr(origOp.kernel_size());
    const auto kernelStrides = parseIntArrayAttr(origOp.strides());

    const auto bitPatternSize =
            VPUIP::NCESparsity::getBitPatternSize(kernelSize, kernelStrides[0], origInputType.getElementType());

    //
    // Prepare input for DPU
    //

    auto inputDPU = prepareInputForDPU(rewriter, origOp->getLoc(), origOp.input());

    //
    // Generate activation window
    //

    const auto fakeSparsity =
            VPUIP::NCESparsity::getFakeSparsity(kernelSize, kernelStrides[0], origInputType.getElementType(), IC);
    const auto activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity, IC);
    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), IC, nullptr, nullptr, activationWindow);

    //
    // Prepare output buffer for DPU
    //

    const auto origOutType = origOp.output().getType().cast<mlir::MemRefType>();
    const auto outReorderType = changeDimsOrder(origOutType, DimsOrder::NHWC);
    const auto outTypeCMX =
            changeMemSpace(outReorderType, VPUIP::PhysicalMemoryAttr::get(getContext(), VPUIP::PhysicalMemory::CMX_NN));

    auto outAllocOpCMX = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), outTypeCMX);

    //
    // Create NCE per-cluster Operation
    //

    const auto padsBegin = parseIntArrayAttr(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr(origOp.pads_end());
    const auto kernelPaddingAttr =
            getInt32ArrayAttr(getContext(), makeArrayRef({padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]}));

    const auto activation_window_channel_length = getInt32Attr(getContext(), static_cast<uint32_t>(bitPatternSize));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), inputDPU, /*weights=*/nullptr, weightsTable, activationWindow,
            /*parent_input=*/inputDPU,
            /*parent_output=*/outAllocOpCMX.memref(),
            /*output_buff=*/outAllocOpCMX.memref(), VPUIP::NCETaskType::MAXPOOL, origOp.kernel_size(), origOp.strides(),
            kernelPaddingAttr, activation_window_channel_length);

    //
    // Create DPU sub-task
    //

    addDPUTasks(nceOp, rewriter, _numDPU, padsBegin, padsEnd, mpeMap.at(_arch));

    //
    // DMA output CMX -> DDR
    //

    rewriter.replaceOpWithNewOp<IERT::CopyOp>(origOp, nceOp.output(), origOp.output_buff());

    return mlir::success();
}

//
// ConvertToNCEOpsPass
//

class ConvertToNCEOpsPass final : public ConvertToNCEOpsBase<ConvertToNCEOpsPass> {
public:
    ConvertToNCEOpsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertToNCEOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = VPUIP::getArch(module);
    VPUX_THROW_UNLESS(mpeMap.find(arch) != mpeMap.end(), "Failed to map MPE mode to target arch");

    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);
    VPUX_THROW_UNLESS(resOp != nullptr, "Missing IERT run-time resources definition");

    auto nceCluster = resOp.getExecutor(VPUIP::PhysicalProcessorAttr::get(&ctx, VPUIP::PhysicalProcessor::NCE_Cluster));
    VPUX_THROW_UNLESS(nceCluster != nullptr, "Failed to get NCE_Cluster information");

    auto dpuExec = nceCluster.getSubExecutor(
            VPUIP::PhysicalProcessorAttr::get(&ctx, VPUIP::PhysicalProcessor::NCE_PerClusterDPU));
    VPUX_THROW_UNLESS(dpuExec != nullptr, "Failed to get DPU information");

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.insert<ConvRewrite>(&ctx, dpuExec.count(), arch, _log);
    patterns.insert<MaxPoolRewrite>(&ctx, dpuExec.count(), arch, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createConvertToNCEOpsPass(Logger log) {
    return std::make_unique<ConvertToNCEOpsPass>(log);
}

//
// Copyright 2021 Intel Corporation.
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
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// Utilities
//

mlir::MemRefType changeMemSpace(mlir::MemRefType origType, mlir::Attribute memSpace) {
    return mlir::MemRefType::Builder(origType).setMemorySpace(memSpace);
}

mlir::MemRefType changeDimsOrder(mlir::MemRefType origType, DimsOrder newOrder) {
    return mlir::MemRefType::Builder(origType).setAffineMaps(newOrder.toAffineMap(origType.getContext()));
}

template <typename T>
mlir::Value createHelperTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<T> data, mlir::Type elemType,
                               ArrayRef<int64_t> shape) {
    auto* ctx = builder.getContext();

    const auto dataStorageType = mlir::RankedTensorType::get(shape, elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, data);

    const auto dataType = mlir::MemRefType::get(shape, elemType);
    auto dataConstOp = builder.create<IERT::ConstantOp>(loc, dataType, dataAttr);

    const auto cmxMemSpaceAttr = VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN);
    const auto dataTypeCMX = changeMemSpace(dataType, cmxMemSpaceAttr);

    auto dataAllocOp = builder.create<mlir::memref::AllocOp>(loc, dataTypeCMX);
    auto copyOp = builder.create<IERT::CopyOp>(loc, dataConstOp.output(), dataAllocOp);

    return copyOp.output();
}

constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

std::vector<int32_t> getWeightsTable(int64_t OC, ConstantInterface biases, int32_t weightPtrStep) {
    const int32_t PRELU_SCALE_OFFSET = 0;
    const int32_t PRELU_SCALE_VALUE = 1;

    // FIXME: PPE shift is actually 6 bit long, 2 higher bits stand for rounding mode
    const int32_t PPE_SHIFT_OFFSET = 8;
    const int32_t PPE_SHIFT_VALUE = 0;

    const int32_t PPE_MULT_OFFSET = 16;
    // FIXME: PPE multiplier has sign, which may affect lower bits
    const int32_t PPE_MULT_VALUE = 1;

    const int32_t mult_shift = (PRELU_SCALE_VALUE << PRELU_SCALE_OFFSET) | (PPE_SHIFT_VALUE << PPE_SHIFT_OFFSET) |
                               (PPE_MULT_VALUE << PPE_MULT_OFFSET);

    const auto getBiasVal = [&](int64_t oc) -> int32_t {
        if (biases == nullptr) {
            return 0;
        }

        const auto biasVals = biases.getContent().getValues<float>();

        // FIXME: 2 ^ 16 might be more obvious
        return std::lround(biasVals[oc] * 65536.f);
    };

    std::vector<int32_t> weightsTableVals(OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, 0);

    // TODO: [Track number: E#13226]
    int32_t weightPtrOffset = 0;

    for (auto oc : irange(checked_cast<size_t>(OC))) {
        const auto wtInd = oc * static_cast<size_t>(WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        weightsTableVals[wtInd + 0] = weightPtrOffset;
        weightsTableVals[wtInd + 1] = 0x0;
        weightsTableVals[wtInd + 2] = mult_shift;
        weightsTableVals[wtInd + 3] = getBiasVal(oc);

        weightPtrOffset += weightPtrStep;
    }

    return weightsTableVals;
}

mlir::Value createWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, int64_t OC, ConstantInterface biases,
                                     int32_t weightPtrStep) {
    const auto weightsTable = getWeightsTable(OC, biases, weightPtrStep);

    SmallVector<int64_t> weightTableShape{OC, 1, 1, WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};

    return createHelperTensor(builder, loc, makeArrayRef(weightsTable), getSInt32Type(builder.getContext()),
                              weightTableShape);
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

void addDPUTasks(VPUIP::NCEClusterTaskOp nceOp, mlir::PatternRewriter& rewriter, const int32_t numDPU,
                 ArrayRef<int64_t> opPadsBegin, ArrayRef<int64_t> opPadsEnd) {
    auto* ctx = nceOp.getContext();
    const auto outputShape = getShape(nceOp.output());
    const auto dpuTiles = VPUIP::DpuTiler::tileOverH(numDPU, outputShape, opPadsBegin, opPadsEnd);

    for (const auto& dpuTile : dpuTiles) {
        const auto startAttr = getInt32ArrayAttr(ctx, makeArrayRef(dpuTile.start));
        const auto endAttr = getInt32ArrayAttr(ctx, makeArrayRef(dpuTile.end));

        const auto padsBeginAttr = getInt32ArrayAttr(ctx, dpuTile.padsBegin);
        const auto padsEndAttr = getInt32ArrayAttr(ctx, dpuTile.padsEnd);

        nceOp.addDPUTask(rewriter, startAttr, endAttr, padsBeginAttr, padsEndAttr, VPUIP::MPEMode::VECTOR_FP16);
    }
}

//
// ConvRewrite
//

class ConvRewrite final : public mlir::OpRewritePattern<IERT::ConvolutionOp> {
public:
    ConvRewrite(mlir::MLIRContext* ctx, uint32_t numDPU, Logger log)
            : mlir::OpRewritePattern<IERT::ConvolutionOp>(ctx), _numDPU(numDPU), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const uint32_t _numDPU;
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
    const auto IC = filterShape[IERT::ConvolutionOp::filter_in_channel_dim()];
    const auto KY = filterShape[IERT::ConvolutionOp::filter_spatial_height_dim()];
    const auto KX = filterShape[IERT::ConvolutionOp::filter_spatial_width_dim()];

    //
    // Prepare filter for DPU
    //

    auto filterDPU = prepareFilterForDPU(rewriter, origOp->getLoc(), origOp.filter());

    //
    // Generate weights table
    //

    ConstantInterface biasConst;
    if (origOp.bias() != nullptr) {
        biasConst = origOp.bias().getDefiningOp<ConstantInterface>();
        VPUX_THROW_UNLESS(biasConst != nullptr, "Only constant biases are supported, got '{0}'", origOp.bias());
    }

    // TODO: [Track number: E#13226]
    // Let's allocate weight table after weights(input),
    // then the weights offset in CMX will be zero
    const auto weightPtrStep = checked_cast<int32_t>(IC * KY * KX * sizeof(int16_t));
    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), OC, biasConst, weightPtrStep);

    //
    // Prepare input for DPU
    //

    auto inputDPU = prepareInputForDPU(rewriter, origOp->getLoc(), origOp.input());

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

    const auto ppeAttr = vpux::VPUIP::PPELayerTypeAttr::get(getContext(), VPUIP::PPELayerType::NOOP);

    const auto padsBegin = parseIntArrayAttr(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr(origOp.pads_end());
    const auto kernelPaddingAttr =
            getInt32ArrayAttr(getContext(), makeArrayRef({padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]}));

    const auto kernelSizeAttr = getInt32ArrayAttr(getContext(), makeArrayRef({KX, KY}));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(origOp->getLoc(), inputDPU, filterDPU, weightsTable, nullptr,
                                                          inputDPU, outAllocOpCMX.memref(), outAllocOpCMX.memref(),
                                                          VPUIP::NCETaskType::CONV, ppeAttr, kernelPaddingAttr,
                                                          origOp.strides(), kernelSizeAttr, nullptr);

    //
    // Create DPU sub-task
    //

    addDPUTasks(nceOp, rewriter, _numDPU, padsBegin, padsEnd);

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
    MaxPoolRewrite(mlir::MLIRContext* ctx, uint32_t numDPU, Logger log)
            : mlir::OpRewritePattern<IERT::MaxPoolOp>(ctx), _numDPU(numDPU), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const uint32_t _numDPU;
    Logger _log;
};

mlir::Value createActivationWindowTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<uint8_t> fakeSparsity,
                                         int64_t numChannels) {
    SmallVector<int64_t> fakeSparsityShape{numChannels, 1, 1, static_cast<int64_t>(fakeSparsity.size()) / numChannels};

    return createHelperTensor(builder, loc, makeArrayRef(fakeSparsity), getUInt8Type(builder.getContext()),
                              fakeSparsityShape);
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
    // Generate activation window
    //

    const auto fakeSparsity =
            VPUIP::NCESparsity::getFakeSparsity(kernelSize, kernelStrides[0], origInputType.getElementType(), IC);
    const auto activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity, IC);

    //
    // Generate weights table
    //

    // TODO: [Track number: E#13226]
    // an activation window offset ??
    // Let's allocate weight table after an activation window,
    // then the an activation window offset in CMX will be zero

    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), IC, nullptr, 0);

    //
    // Prepare input for DPU
    //

    auto inputDPU = prepareInputForDPU(rewriter, origOp->getLoc(), origOp.input());

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

    const auto ppeAttr = vpux::VPUIP::PPELayerTypeAttr::get(getContext(), VPUIP::PPELayerType::NOOP);

    const auto padsBegin = parseIntArrayAttr(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr(origOp.pads_end());
    const auto kernelPaddingAttr =
            getInt32ArrayAttr(getContext(), makeArrayRef({padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]}));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), inputDPU, nullptr, weightsTable, activationWindow, inputDPU, outAllocOpCMX.memref(),
            outAllocOpCMX.memref(), VPUIP::NCETaskType::MAXPOOL, ppeAttr, kernelPaddingAttr, origOp.strides(),
            origOp.kernel_size(), getInt32Attr(getContext(), static_cast<uint32_t>(bitPatternSize)));

    //
    // Create DPU sub-task
    //

    addDPUTasks(nceOp, rewriter, _numDPU, padsBegin, padsEnd);

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

    auto resOp = IERT::RunTimeResourcesOp::getFromModule(func->getParentOfType<mlir::ModuleOp>());
    VPUX_THROW_UNLESS(resOp != nullptr, "Missing IERT run-time resources definition");

    auto nceCluster = resOp.getExecutor(VPUIP::PhysicalProcessorAttr::get(&ctx, VPUIP::PhysicalProcessor::NCE_Cluster));
    VPUX_THROW_UNLESS(nceCluster != nullptr, "Failed to get NCE_Cluster information");

    auto dpuExec = nceCluster.getSubExecutor(
            VPUIP::PhysicalProcessorAttr::get(&ctx, VPUIP::PhysicalProcessor::NCE_PerClusterDPU));
    VPUX_THROW_UNLESS(dpuExec != nullptr, "Failed to get DPU information");

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.insert<ConvRewrite>(&ctx, dpuExec.count(), _log);
    patterns.insert<MaxPoolRewrite>(&ctx, dpuExec.count(), _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createConvertToNCEOpsPass(Logger log) {
    return std::make_unique<ConvertToNCEOpsPass>(log);
}

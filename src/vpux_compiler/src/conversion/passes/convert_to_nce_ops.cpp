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

std::tuple<mlir::ArrayAttr, mlir::ArrayAttr> getDPUTaskCoords(mlir::MLIRContext* ctx, ShapeRef shape) {
    VPUX_THROW_UNLESS(shape.size() == 4, "getDPUTaskCoords works with 4-d tensors only");

    const int32_t C = checked_cast<int32_t>(shape[IERT::ConvolutionOp::act_channel_dim()]);
    const int32_t H = checked_cast<int32_t>(shape[IERT::ConvolutionOp::act_height_dim()]);
    const int32_t W = checked_cast<int32_t>(shape[IERT::ConvolutionOp::act_width_dim()]);

    const auto startAttr = getInt32ArrayAttr(ctx, makeArrayRef({0, 0, 0}));
    const auto endAttr = getInt32ArrayAttr(ctx, makeArrayRef({W - 1, H - 1, C - 1}));

    return std::make_tuple(startAttr, endAttr);
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
    const auto origType = input.getType().cast<mlir::MemRefType>();

    // reorder NCHW -> NHWC
    const auto typeNHWC = changeDimsOrder(origType, DimsOrder::NHWC);
    auto reorderAllocOp = builder.create<mlir::memref::AllocOp>(loc, typeNHWC);
    auto reorderOp = builder.create<IERT::ReorderOp>(loc, input, reorderAllocOp.memref());

    // DMA DDR -> CMX
    auto typeCMX = changeMemSpace(typeNHWC,
                                  VPUIP::PhysicalMemoryAttr::get(builder.getContext(), VPUIP::PhysicalMemory::CMX_NN));
    auto dmaAllocOp = builder.create<mlir::memref::AllocOp>(loc, typeCMX);
    auto dmaOp = builder.create<IERT::CopyOp>(loc, reorderOp.output(), dmaAllocOp.memref());

    return dmaOp.output();
}

//
// ConvRewrite
//

class ConvRewrite final : public mlir::OpRewritePattern<IERT::ConvolutionOp> {
public:
    ConvRewrite(mlir::MLIRContext* ctx, Byte cmxSize, Logger log)
            : mlir::OpRewritePattern<IERT::ConvolutionOp>(ctx), _cmxSize(cmxSize), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Byte _cmxSize;
    Logger _log;
};

mlir::Value prepareFilterForDPU(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value filter) {
    const auto origType = filter.getType().cast<mlir::MemRefType>();

    // reorder
    const auto deviceFilterOrder = DimsOrder::fromPermutation({
            IERT::ConvolutionOp::filter_out_channel_dim(),     //
            IERT::ConvolutionOp::filter_spatial_height_dim(),  //
            IERT::ConvolutionOp::filter_spatial_width_dim(),   //
            IERT::ConvolutionOp::filter_in_channel_dim()       //
    });
    const auto reorderType = changeDimsOrder(origType, deviceFilterOrder);
    auto reorderAllocOp = builder.create<mlir::memref::AllocOp>(loc, reorderType);
    auto reorderOp = builder.create<IERT::ReorderOp>(loc, filter, reorderAllocOp.memref());

    // DMA DDR -> CMX
    auto typeCMX = changeMemSpace(reorderType,
                                  VPUIP::PhysicalMemoryAttr::get(builder.getContext(), VPUIP::PhysicalMemory::CMX_NN));
    auto dmaAllocOp = builder.create<mlir::memref::AllocOp>(loc, typeCMX);
    auto dmaOp = builder.create<IERT::CopyOp>(loc, reorderOp.output(), dmaAllocOp.memref());

    return dmaOp.output();
}

mlir::LogicalResult ConvRewrite::matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    //
    // Get dimensions
    //

    const auto filterShape = getShape(origOp.filter());

    const auto OC = filterShape[IERT::ConvolutionOp::filter_out_channel_dim()];
    const auto IC = filterShape[IERT::ConvolutionOp::filter_in_channel_dim()];
    const auto KY = filterShape[IERT::ConvolutionOp::filter_spatial_height_dim()];
    const auto KX = filterShape[IERT::ConvolutionOp::filter_spatial_width_dim()];

    if (KX > 11 || KY > 11) {
        return matchFailed(rewriter, origOp, "Unsupported kernel size {0}x{1}. Supported size up to 11", KX, KY);
    }

    //
    // Check channel alignment
    //

    static constexpr int64_t CHANNEL_ALIGNMENT = 16;

    if (OC % CHANNEL_ALIGNMENT != 0) {
        return matchFailed(rewriter, origOp, "Output channels are not aligned");
    }
    if (IC % CHANNEL_ALIGNMENT != 0) {
        return matchFailed(rewriter, origOp, "Input channels are not aligned");
    }

    //
    // Check that buffers fit into CMX
    //

    Byte requiredCMX(0);

    for (const auto& operand : origOp.getOpOperands()) {
        requiredCMX += getTotalSize(operand.get());
    }

    requiredCMX += OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * 4_Byte;

    if (requiredCMX > _cmxSize) {
        return matchFailed(rewriter, origOp, "CMX memory is not enough, available '{0}', required '{1}'", _cmxSize,
                           requiredCMX);
    }

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

    mlir::ArrayAttr startAttr, endAttr;
    std::tie(startAttr, endAttr) = getDPUTaskCoords(getContext(), getShape(nceOp.output()));

    const auto padsBeginAttr = getInt32ArrayAttr(getContext(), padsBegin);
    const auto padsEndAttr = getInt32ArrayAttr(getContext(), padsEnd);

    nceOp.addDPUTask(rewriter, startAttr, endAttr, padsBeginAttr, padsEndAttr, VPUIP::MPEMode::VECTOR_FP16);

    //
    // DMA output CMX -> DDR
    //

    auto outAllocOpDDR = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), outReorderType);
    auto outDMAOp = rewriter.create<IERT::CopyOp>(origOp->getLoc(), nceOp.output(), outAllocOpDDR.memref());

    //
    // Reorder output NHWC -> NCHW
    //

    rewriter.replaceOpWithNewOp<IERT::ReorderOp>(origOp, outDMAOp.output(), origOp.output_buff());

    return mlir::success();
}

//
// MaxPoolRewrite
//

class MaxPoolRewrite final : public mlir::OpRewritePattern<IERT::MaxPoolOp> {
public:
    MaxPoolRewrite(mlir::MLIRContext* ctx, Byte cmxSize, Logger log)
            : mlir::OpRewritePattern<IERT::MaxPoolOp>(ctx), _cmxSize(cmxSize), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Byte _cmxSize;
    Logger _log;
};

int64_t getWindowSize(int64_t kernelW, int64_t strideW, mlir::Type elemType) {
    // Select the maximum window size not exceeding 32 bytes
    // by iterating through the MPE_NUM values (2, 4, 8, 16)

    VPUX_THROW_UNLESS(elemType.isInteger(8) || elemType.isF16(), "Supported only I8 and FP16 types");

    // Only MPE0, MPE4, MPE8 and MPE12 support FP16 data format
    const int mpeNumLimit = elemType.isF16() ? 4 : 16;
    const int64_t maxWindowSize = 32;

    int64_t windowSize = 0;
    int mpeNum = 2;

    while (mpeNum <= mpeNumLimit) {
        if (strideW <= kernelW) {
            windowSize = kernelW + strideW * (mpeNum - 1);
        } else {
            windowSize = kernelW * mpeNum;
        }

        if (windowSize > maxWindowSize)
            return windowSize;

        mpeNum *= 2;
    }

    return windowSize;
}

std::vector<uint8_t> getBitPattern(ArrayRef<int64_t> kernelSize, int64_t windowSize) {
    const auto kernelW = kernelSize[0];
    const auto kernelH = kernelSize[1];

    VPUX_THROW_UNLESS(windowSize >= kernelW,
                      "windowsSize must be greater than or equal to kernelW. windowsSize={0}, kernelW={1}", windowSize,
                      kernelW);

    const auto numBitsSet = kernelW;
    const auto numBitsClear = windowSize - kernelW;

    SmallVector<uint8_t> window;
    window.reserve(windowSize);
    window.insert(window.end(), numBitsSet, 1);
    window.insert(window.end(), numBitsClear, 0);

    const auto numOfRepeat = kernelH;

    std::vector<uint8_t> bitPattern;
    bitPattern.reserve(numOfRepeat * windowSize);
    for (auto i = 0; i < numOfRepeat; i++) {
        bitPattern.insert(bitPattern.end(), window.begin(), window.end());
    }

    return bitPattern;
}

std::vector<uint8_t> getFakeSparsity(ArrayRef<uint8_t> bitPattern, int64_t inputChannels) {
    // To align each activation map entry to 16 bytes to abide the hw restriction
    const auto perChannelSparsitySize = static_cast<std::size_t>(std::ceil(bitPattern.size() / 128.0) * 16);

    // MaxPool is supported only in depth wise mode.
    // Depth wise does not support weights sparsity in the real sense,
    // but it will have to have an activation window pointer,
    // which is regarded as "fake sparsity"
    SmallVector<uint8_t> perChannelSparsity;
    perChannelSparsity.resize(perChannelSparsitySize);

    // Repackaging each byte from bitPattern to a bit from fakeSparsity
    // The rest of the bits remain zero
    for (size_t i = 0; i < bitPattern.size(); i++) {
        perChannelSparsity[(i / 128) * 16 + (i % 128) / 8] |= bitPattern[i] << (i % 8);
    }

    std::vector<uint8_t> fakeSparsity;
    fakeSparsity.reserve(inputChannels * perChannelSparsitySize);
    for (auto i = 0; i < inputChannels; i++) {
        fakeSparsity.insert(fakeSparsity.end(), perChannelSparsity.begin(), perChannelSparsity.end());
    }

    return fakeSparsity;
}

mlir::Value createActivationWindowTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<int64_t> kernelSize,
                                         int64_t numChannels, int64_t windowSize) {
    const auto bitPattern = getBitPattern(kernelSize, windowSize);
    const auto fakeSparsity = getFakeSparsity(bitPattern, numChannels);

    SmallVector<int64_t> fakeSparsityShape{numChannels, 1, 1, static_cast<int64_t>(fakeSparsity.size()) / numChannels};

    return createHelperTensor(builder, loc, makeArrayRef(fakeSparsity), getUInt8Type(builder.getContext()),
                              fakeSparsityShape);
}

mlir::LogicalResult MaxPoolRewrite::matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    //
    // Get dimensions
    //

    const auto origInputType = origOp.input().getType().cast<mlir::MemRefType>();
    const auto inputShape = getShape(origInputType);

    const auto IC = inputShape[IERT::MaxPoolOp::act_channel_dim()];

    const auto kernelSize = parseIntArrayAttr(origOp.kernel_size());
    const auto kernelStrides = parseIntArrayAttr(origOp.strides());

    VPUX_THROW_UNLESS(kernelSize.size() == 2, "Unsupported kernel size: %d", kernelSize.size());
    VPUX_THROW_UNLESS(kernelStrides.size() == 2, "Unsupported strides size: %d", kernelSize.size());

    if (kernelSize[0] > 11 || kernelSize[1] > 11) {
        return matchFailed(rewriter, origOp, "Unsupported kernel size {0}x{1}. Supported size up to 11", kernelSize[0],
                           kernelSize[1]);
    }

    const auto windowSize = getWindowSize(kernelSize[0], kernelStrides[0], origInputType.getElementType());
    const auto bitPatternSize = kernelSize[1] * windowSize;
    const auto perChannelSparsitySize = static_cast<std::size_t>(std::ceil(bitPatternSize / 128.0) * 16);
    const auto activationWindowSize = IC * perChannelSparsitySize;

    //
    // Check channel alignment
    //

    static constexpr int64_t CHANNEL_ALIGNMENT = 16;

    if (IC % CHANNEL_ALIGNMENT != 0) {
        return matchFailed(rewriter, origOp, "Input channels are not aligned");
    }

    //
    // Check that buffers fit into CMX
    //

    Byte requiredCMX(0);

    for (const auto& operand : origOp.getOpOperands()) {
        requiredCMX += getTotalSize(operand.get());
    }

    requiredCMX += IC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * 4_Byte;

    requiredCMX += activationWindowSize * 1_Byte;

    if (requiredCMX > _cmxSize) {
        return matchFailed(rewriter, origOp, "CMX memory is not enough, available '{0}', required '{1}'", _cmxSize,
                           requiredCMX);
    }

    //
    // Generate activation window
    //

    const auto activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), kernelSize, IC, windowSize);

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

    mlir::ArrayAttr startAttr, endAttr;
    std::tie(startAttr, endAttr) = getDPUTaskCoords(getContext(), getShape(nceOp.output()));

    const auto padsBeginAttr = getInt32ArrayAttr(getContext(), padsBegin);
    const auto padsEndAttr = getInt32ArrayAttr(getContext(), padsEnd);

    nceOp.addDPUTask(rewriter, startAttr, endAttr, padsBeginAttr, padsEndAttr, VPUIP::MPEMode::VECTOR_FP16);

    //
    // DMA output CMX -> DDR
    //

    auto outAllocOpDDR = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), outReorderType);
    auto outDMAOp = rewriter.create<IERT::CopyOp>(origOp->getLoc(), nceOp.output(), outAllocOpDDR.memref());

    //
    // Reorder output NHWC -> NCHW
    //

    rewriter.replaceOpWithNewOp<IERT::ReorderOp>(origOp, outDMAOp.output(), origOp.output_buff());

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
    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);

    auto cmxSize = resOp.getAvailableMemory(VPUIP::PhysicalMemoryAttr::get(&ctx, VPUIP::PhysicalMemory::CMX_NN));
    VPUX_THROW_UNLESS(cmxSize != nullptr, "Can't get information about {0} memory", VPUIP::PhysicalMemory::CMX_NN);

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.insert<ConvRewrite>(&ctx, cmxSize.size(), _log);
    patterns.insert<MaxPoolRewrite>(&ctx, cmxSize.size(), _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createConvertToNCEOpsPass(Logger log) {
    return std::make_unique<ConvertToNCEOpsPass>(log);
}

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
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ConvertToNCEOpsPass
//

class ConvertToNCEOpsPass final : public ConvertToNCEOpsBase<ConvertToNCEOpsPass> {
public:
    ConvertToNCEOpsPass(Logger log);

public:
    class MaxPoolRewrite;
    class ConvRewrite;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

ConvertToNCEOpsPass::ConvertToNCEOpsPass(Logger log): _log(log) {
    _log.setName(Base::getArgumentName());
}

static std::tuple<mlir::ArrayAttr, mlir::ArrayAttr> getDPUTaskCoords(mlir::MLIRContext* ctx, ShapeRef shape) {
    VPUX_THROW_UNLESS(shape.size() == 4, "getDPUTaskCoords works with 4-d tensors only");
    Dim C = Dim(1), H = Dim(2), W = Dim(3);
    SmallVector<int32_t> start = {0, 0, 0};
    // subtract one due to the runtime specific
    SmallVector<int32_t> end = {static_cast<int32_t>(shape[W] - 1), static_cast<int32_t>(shape[H] - 1),
                                static_cast<int32_t>(shape[C] - 1)};
    auto startAttr = getInt32ArrayAttr(ctx, start);
    auto endAttr = getInt32ArrayAttr(ctx, end);
    return std::make_tuple(startAttr, endAttr);
}

static mlir::MemRefType buildMemSpaceHelper(mlir::MemRefType origType, mlir::Attribute memSpace) {
    return mlir::MemRefType::Builder(origType).setMemorySpace(memSpace);
}

template <typename T>
static mlir::Value createHelperTensor(LayerInterface layer, mlir::OpBuilder& builder, ArrayRef<T> data, mlir::Type type,
                                      ArrayRef<int64_t> shape) {
    auto ctx = layer.getContext();

    const auto dataType = mlir::MemRefType::get(shape, type);
    const auto dataStorageType = mlir::RankedTensorType::get(shape, type);

    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, data);
    auto dataConstOp = builder.create<IERT::ConstantOp>(layer->getLoc(), dataType, dataAttr);

    auto cmxMemSpaceAttr = VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN);
    auto newDataTypeCMX = buildMemSpaceHelper(dataType, cmxMemSpaceAttr);

    auto dataAllocOp = builder.create<mlir::memref::AllocOp>(dataConstOp->getLoc(), newDataTypeCMX);
    auto copyOp = builder.create<IERT::CopyOp>(dataConstOp->getLoc(), dataConstOp.output(), dataAllocOp);

    return copyOp.output();
}

static mlir::Value createWeightsTable(LayerInterface layer, mlir::OpBuilder& builder,
                                      const std::vector<int32_t>& weightTableVals) {
    auto outputs = layer.getOutputs();

    Dim C = Dim(1);
    const auto outputShape = getShape(outputs[0]);
    auto numChannelElements = outputShape[C];
    SmallVector<int64_t> weightTableShape{numChannelElements, 1, 1, 4};

    auto ctx = layer.getContext();
    return createHelperTensor(layer, builder, makeArrayRef(weightTableVals), getSInt32Type(ctx), weightTableShape);
}

static mlir::Value createWeights(LayerInterface opLayer, mlir::OpBuilder& builder, const mlir::Value& filter) {
    auto ctx = opLayer.getContext();

    const Dim Out(0);
    const Dim In(1);
    const Dim KernelY(2);
    const Dim KernelX(3);
    const DimsOrder deviceFilterOrder = DimsOrder::fromPermutation({Out, KernelY, KernelX, In});
    auto weightsType_nchw = filter.getType().dyn_cast<mlir::MemRefType>();
    auto weightsType_nhwc =
            mlir::MemRefType::get(weightsType_nchw.getShape(), weightsType_nchw.getElementType(),
                                  deviceFilterOrder.toAffineMap(ctx), weightsType_nchw.getMemorySpace());
    auto weights_nhwc_AllocOp = builder.create<mlir::memref::AllocOp>(opLayer->getLoc(), weightsType_nhwc);
    auto weights_nhwc_ddr = builder.create<IERT::ReorderOp>(opLayer->getLoc(), filter, weights_nhwc_AllocOp);

    auto weightsType = weights_nhwc_ddr.getType().dyn_cast<mlir::MemRefType>();
    auto cmxMemSpaceAttr = VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN);
    auto newTypeWeightsCMX = buildMemSpaceHelper(weightsType, cmxMemSpaceAttr);

    auto weightsCMXAllocOp = builder.create<mlir::memref::AllocOp>(opLayer->getLoc(), newTypeWeightsCMX);
    auto weightsCMXAllocOpCopy = builder.create<IERT::CopyOp>(opLayer->getLoc(), weights_nhwc_ddr, weightsCMXAllocOp);

    return weightsCMXAllocOpCopy;
}

std::vector<int32_t> generateWTablesValues(const size_t channelSize, const int32_t weightPtrStep,
                                           const std::vector<int32_t>& biases) {
    const int32_t PRELU_SCALE_OFFSET = 0;
    const int32_t PRELU_SCALE_VALUE = 1;

    // FIXME PPE shift is actually 6 bit long, 2 higher bits stand for rounding mode
    const int32_t PPE_SHIFT_OFFSET = 8;
    const int32_t PPE_SHIFT_VALUE = 0;

    const int32_t PPE_MULT_OFFSET = 16;
    // FIXME PPE multiplier has sign, which may affect lower bits
    const int32_t PPE_MULT_VALUE = 1;

    const int32_t mult_shift = (PRELU_SCALE_VALUE << PRELU_SCALE_OFFSET) | (PPE_SHIFT_VALUE << PPE_SHIFT_OFFSET) |
                               (PPE_MULT_VALUE << PPE_MULT_OFFSET);  // 0x00010001

    std::vector<int32_t> weightsTableVals(channelSize * 4, 0);
    int32_t weightPtrOffset = 0;  // TODO: [Track number: E#13226]
    for (size_t i = 0; i < weightsTableVals.size(); i += 4) {
        const int32_t PPE_BIAS_VALUE = biases.at(i / 4);

        weightsTableVals.at(i + 0) = weightPtrOffset;
        weightsTableVals.at(i + 1) = 0x0;
        weightsTableVals.at(i + 2) = mult_shift;
        weightsTableVals.at(i + 3) = PPE_BIAS_VALUE;

        weightPtrOffset += weightPtrStep;
    }

    return weightsTableVals;
}

//
// ConvRewrite
//

class ConvertToNCEOpsPass::ConvRewrite final : public mlir::OpRewritePattern<IERT::ConvolutionOp> {
public:
    ConvRewrite(mlir::MLIRContext* ctx, Logger log, IERT::RunTimeResourcesOp resources)
            : mlir::OpRewritePattern<IERT::ConvolutionOp>(ctx), _log(log), _resources(resources) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    mutable IERT::RunTimeResourcesOp _resources;
};

mlir::LogicalResult ConvertToNCEOpsPass::ConvRewrite::matchAndRewrite(IERT::ConvolutionOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto cmxMemSpaceAttr = VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN);
    size_t availableNNCMXSizeInBytes = _resources.getAvailableMemory(cmxMemSpaceAttr).byteSize();
    size_t requiredNNCMXBytes = 0;
    for (auto val : {origOp.filter(), origOp.input(), origOp.output()}) {
        auto type = val.getType().cast<mlir::MemRefType>();
        requiredNNCMXBytes += getTypeTotalSize(type).count();
    }
    static const size_t NUM_ELEMENTS_PER_WEIGHT_TABLE_ROW = 4;
    requiredNNCMXBytes += origOp.filter().getType().cast<mlir::ShapedType>().getShape().front() *
                          NUM_ELEMENTS_PER_WEIGHT_TABLE_ROW * 4;

    if (requiredNNCMXBytes > availableNNCMXSizeInBytes / 4) {
        return matchFailed(rewriter, origOp, "CMX memory is not enough");
    }

    for (const auto& operand : origOp.getOpOperands()) {
        auto shape = operand.get().getType().cast<mlir::ShapedType>().getShape();  // TODO: Fix this assumption.
        constexpr int CHANNEL_ALIGNMENT = 16;
        if (shape[1] % CHANNEL_ALIGNMENT != 0) {
            return matchFailed(rewriter, origOp, "Channels are not aligned");
        }
    }

    // prepare weights in CMX
    const auto dim_c = Dim(1);
    const auto dim_h = Dim(2);
    const auto dim_w = Dim(3);
    const auto filter_shape = getShape(origOp.filter());

    auto weightsCMXAllocOpCopy = createWeights(origOp, rewriter, origOp.filter());

    auto padsBegin = parseIntArrayAttr(origOp.pads_begin());
    auto padsEnd = parseIntArrayAttr(origOp.pads_end());

    auto padsBeginAttr = getInt32ArrayAttr(ctx, padsBegin);
    auto padsEndAttr = getInt32ArrayAttr(ctx, padsEnd);

    auto kernelPaddingAttr = getInt32ArrayAttr(ctx, makeArrayRef({padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]}));

    const auto outputShape = getShape(origOp.output().getType().cast<mlir::ShapedType>());
    std::vector<int32_t> biases(outputShape[dim_c]);

    if (origOp.bias()) {
        auto biasConst = origOp.bias().getDefiningOp<ConstantInterface>();
        for (auto p : enumerate(biasConst.getContent().getValues<float>())) {
            int32_t biasVal = std::lround(p.value() * 65536.f);  // FIXME 2 ^ 16 might be more obvious
            biases.at(p.index()) = biasVal;
        }
    }

    // Generate weights table

    // TODO: [Track number: E#13226]
    // Lets allocate weight table after weights(input),
    // then the weights offset in CMX will be zero
    int32_t weightPtrStep = static_cast<int32_t>(filter_shape[dim_c]) * static_cast<int32_t>(filter_shape[dim_h]) *
                            static_cast<int32_t>(filter_shape[dim_w]) * sizeof(int16_t);
    std::vector<int32_t> weightTableVals =
            generateWTablesValues(static_cast<int32_t>(outputShape[dim_c]), weightPtrStep, biases);

    auto weightsTableAllocOpCopy = createWeightsTable(origOp, rewriter, weightTableVals);

    //                              memref (null) -> NCEOp -> memref (null)
    // memref (null) -> CopyOp -> memref (CMX_NN) -> NCEOp -> memref (CMX_NN) -> CopyOp -> memref (null)

    // NCHW -> NHWC for input
    const DimsOrder dimsOrderZMajor = DimsOrder::NHWC;

    auto inputType_nchw = origOp.input().getType().dyn_cast<mlir::MemRefType>();
    auto inputType_nhwc = mlir::MemRefType::get(inputType_nchw.getShape(), inputType_nchw.getElementType(),
                                                dimsOrderZMajor.toAffineMap(ctx), inputType_nchw.getMemorySpace());

    auto input_nhwc_AllocOp = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), inputType_nhwc);
    auto input_nhwc_ddr = rewriter.create<IERT::ReorderOp>(origOp->getLoc(), origOp.input(), input_nhwc_AllocOp);

    auto inputType = input_nhwc_ddr.getType().dyn_cast<mlir::MemRefType>();
    // prepare input in CMX
    auto newTypeInputCMX = buildMemSpaceHelper(inputType, cmxMemSpaceAttr);
    auto inputCMXAllocOp = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), newTypeInputCMX);
    auto inputCMXAllocOpCopy = rewriter.create<IERT::CopyOp>(origOp->getLoc(), input_nhwc_ddr, inputCMXAllocOp);
    // end input processing

    // prepare output in CMX
    auto outputType = origOp.output().getType().dyn_cast<mlir::MemRefType>();
    auto newTypeOutputCMX =
            mlir::MemRefType::get(outputType.getShape(), outputType.getElementType(), dimsOrderZMajor.toAffineMap(ctx),
                                  VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN));

    auto outputCMXAllocOp = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), newTypeOutputCMX);

    auto ppeAttr = vpux::VPUIP::PPELayerTypeAttr::get(ctx, VPUIP::PPELayerType::NOOP);

    mlir::ArrayAttr startAttr, endAttr;
    const auto shape = getShape(origOp.output().getType().cast<mlir::ShapedType>());
    std::tie(startAttr, endAttr) = getDPUTaskCoords(ctx, shape);

    const auto filterShape = getShape(origOp.filter());

    static const auto KH = Dim(2);
    static const auto KW = Dim(3);
    const SmallVector<int64_t> start = {filterShape[KW], filterShape[KH]};
    mlir::ArrayAttr kernelSize = getInt32ArrayAttr(ctx, start);

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), inputCMXAllocOpCopy, weightsCMXAllocOpCopy, weightsTableAllocOpCopy, nullptr,
            inputCMXAllocOpCopy, outputCMXAllocOp, outputCMXAllocOp, VPUIP::NCETaskType::CONV, ppeAttr,
            kernelPaddingAttr, origOp.strides(), kernelSize, nullptr);

    nceOp.addDPUTask(rewriter, startAttr, endAttr, padsBeginAttr, padsEndAttr, VPUIP::MPEMode::VECTOR_FP16);

    // CMX -> DDR
    auto cmx_nhwc_mem_ref = nceOp.output().getType().dyn_cast<mlir::MemRefType>();
    auto ddr_nhwc_type = mlir::MemRefType::get(cmx_nhwc_mem_ref.getShape(), cmx_nhwc_mem_ref.getElementType(),
                                               cmx_nhwc_mem_ref.getAffineMaps(),
                                               origOp.output().getType().dyn_cast<mlir::MemRefType>().getMemorySpace());
    auto ddr_nhwc_alloc_op = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), ddr_nhwc_type);
    auto output_ddr_nhwc = rewriter.create<IERT::CopyOp>(origOp->getLoc(), nceOp.output(), ddr_nhwc_alloc_op);

    // NHWC -> NCHW
    auto outputReorderOp = rewriter.create<IERT::ReorderOp>(origOp->getLoc(), output_ddr_nhwc, origOp.output_buff());

    rewriter.replaceOp(origOp, outputReorderOp->getResults());

    return mlir::success();
}

//
// MaxPoolRewrite
//

static int64_t getWindowSize(int64_t kernelW, int64_t strideW, mlir::Type dataType) {
    // Select the maximum window size not exceeding 32 bytes
    // by iterating through the MPE_NUM values (2, 4, 8, 16)

    VPUX_THROW_UNLESS(dataType.isInteger(8) || dataType.isF16(), "Supported only I8 and FP16 types");

    // Only MPE0, MPE4, MPE8 and MPE12 support FP16 data format
    const auto mpeNumLimit = dataType.isF16() ? 4 : 16;
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

std::vector<uint8_t> getBitPattern(ArrayRef<int64_t> kernelSize, int64_t windowSize, int64_t inputChannels) {
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

    const auto numOfRepeat = kernelH * inputChannels;

    std::vector<uint8_t> bitPattern;
    bitPattern.reserve(numOfRepeat * windowSize);
    for (auto i = 0; i < numOfRepeat; i++) {
        bitPattern.insert(bitPattern.end(), window.begin(), window.end());
    }

    return bitPattern;
}

std::vector<uint8_t> getFakeSparsity(ArrayRef<uint8_t> bitPattern, int64_t inputChannels) {
    // to align each activation map entry to 16 bytes to abide the hw restriction
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

static mlir::Value createActivationWindow(LayerInterface layer, mlir::OpBuilder& builder,
                                          const std::vector<uint8_t>& fakeSparsity) {
    auto inputs = layer.getInputs();
    Dim C = Dim(1);
    const auto inputShape = getShape(inputs[0]);
    auto numChannelElements = inputShape[C];
    SmallVector<int64_t> fakeSparsityShape{numChannelElements, 1, 1,
                                           static_cast<int64_t>(fakeSparsity.size()) / numChannelElements};

    auto ctx = layer.getContext();
    return createHelperTensor(layer, builder, makeArrayRef(fakeSparsity), getUInt8Type(ctx), fakeSparsityShape);
}

class ConvertToNCEOpsPass::MaxPoolRewrite final : public mlir::OpRewritePattern<IERT::MaxPoolOp> {
public:
    MaxPoolRewrite(mlir::MLIRContext* ctx, Logger log, IERT::RunTimeResourcesOp resources)
            : mlir::OpRewritePattern<IERT::MaxPoolOp>(ctx), _log(log), _resources(resources) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    mutable IERT::RunTimeResourcesOp _resources;
};

mlir::LogicalResult ConvertToNCEOpsPass::MaxPoolRewrite::matchAndRewrite(IERT::MaxPoolOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    size_t availableNNCMXSizeInBytes =
            _resources.getAvailableMemory(VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN))
                    .byteSize();
    size_t requiredNNCMXBytes = 0;
    for (const auto& operand : origOp.getOpOperands()) {
        auto type = operand.get().getType().cast<mlir::MemRefType>();
        requiredNNCMXBytes += getTypeTotalSize(type).count();
    }

    if (requiredNNCMXBytes > availableNNCMXSizeInBytes / 4) {
        return matchFailed(rewriter, origOp, "CMX memory is not enough");
    }

    for (const auto& operand : origOp.getOpOperands()) {
        auto shape = operand.get().getType().cast<mlir::ShapedType>().getShape();  // TODO: Fix this assumption.
        constexpr int CHANNEL_ALIGNMENT = 16;
        if (shape[1] % CHANNEL_ALIGNMENT != 0) {
            return matchFailed(rewriter, origOp, "Channels are not aligned");
        }
    }

    auto inputTypeNCHW = origOp.input().getType().dyn_cast<mlir::MemRefType>();

    const DimsOrder dimsOrderZMajor = DimsOrder::NHWC;
    auto reorderedInputType = mlir::MemRefType::get(inputTypeNCHW.getShape(), inputTypeNCHW.getElementType(),
                                                    dimsOrderZMajor.toAffineMap(ctx), inputTypeNCHW.getMemorySpace());

    auto reorderedInputAllocOp = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), reorderedInputType);
    auto reorderedInput = rewriter.create<IERT::ReorderOp>(origOp->getLoc(), origOp.input(), reorderedInputAllocOp);

    auto padsBegin = parseIntArrayAttr(origOp.pads_begin());
    auto padsEnd = parseIntArrayAttr(origOp.pads_end());

    auto padsBeginAttr = getInt32ArrayAttr(ctx, padsBegin);
    auto padsEndAttr = getInt32ArrayAttr(ctx, padsEnd);

    auto kernelPaddingAttr = getInt32ArrayAttr(ctx, makeArrayRef({padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]}));

    Dim C = Dim(1);
    const auto outputShape = getShape(origOp.output().getType().cast<mlir::ShapedType>());
    const auto inputShape = getShape(origOp.input().getType().cast<mlir::ShapedType>());

    // Generate activation window

    const auto strides = parseIntArrayAttr(origOp.strides());
    const auto kernelSize = parseIntArrayAttr(origOp.kernel_size());

    VPUX_THROW_UNLESS(strides.size() == 2, "Unsupported strides size: %d", kernelSize.size());
    VPUX_THROW_UNLESS(kernelSize.size() == 2, "Unsupported kernel size: %d", kernelSize.size());
    VPUX_THROW_UNLESS(kernelSize[0] <= 11 && kernelSize[1] <= 11,
                      "Unsupported kernel size {0}x{1}. Supported size up to 11", kernelSize[0], kernelSize[1]);

    const auto windowSize = getWindowSize(kernelSize[0], strides[0], inputTypeNCHW.getElementType());
    const auto bitPattern = getBitPattern(kernelSize, windowSize, 1);
    const auto fakeSparsity = getFakeSparsity(bitPattern, inputShape[C]);

    auto activationWindowAllocOp = createActivationWindow(origOp, rewriter, fakeSparsity);

    // Generate weights table

    // an activation window offset ??
    // TODO: [Track number: E#13226]
    // Lets allocate weight table after an activation window,
    // then the an activation window offset in CMX will be zero
    std::vector<int32_t> biases(outputShape[C], 0);
    std::vector<int32_t> weightTableVals = generateWTablesValues(outputShape[C], 0, biases);

    auto weightsTableAllocOp = createWeightsTable(origOp, rewriter, weightTableVals);

    //
    //                              memref (null) -> NCEOp -> memref (null)
    // memref (null) -> CopyOp -> memref (CMX_NN) -> NCEOp -> memref (CMX_NN) -> CopyOp -> memref (null)

    // prepare input in CMX
    auto inputType = reorderedInput.getType().dyn_cast<mlir::MemRefType>();
    auto newTypeInputCMX =
            mlir::MemRefType::get(inputType.getShape(), inputType.getElementType(), inputType.getAffineMaps(),
                                  VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN));
    auto inputCMXAllocOp = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), newTypeInputCMX);
    auto inputCMXAllocOpCopy = rewriter.create<IERT::CopyOp>(origOp->getLoc(), reorderedInput, inputCMXAllocOp);
    // end input processing

    // prepare output in CMX
    auto outputType = origOp.output().getType().dyn_cast<mlir::MemRefType>();
    auto newTypeOutputCMX =
            mlir::MemRefType::get(outputType.getShape(), outputType.getElementType(), dimsOrderZMajor.toAffineMap(ctx),
                                  VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN));

    auto outputCMXAllocOp = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), newTypeOutputCMX);

    auto ppeAttr = vpux::VPUIP::PPELayerTypeAttr::get(ctx, VPUIP::PPELayerType::NOOP);

    mlir::ArrayAttr startAttr, endAttr;
    const auto shape = getShape(origOp.output().getType().cast<mlir::ShapedType>());
    std::tie(startAttr, endAttr) = getDPUTaskCoords(ctx, shape);

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), inputCMXAllocOpCopy, nullptr, weightsTableAllocOp, activationWindowAllocOp,
            inputCMXAllocOpCopy, outputCMXAllocOp, outputCMXAllocOp, VPUIP::NCETaskType::MAXPOOL, ppeAttr,
            kernelPaddingAttr, origOp.strides(), origOp.kernel_size(),
            getInt32Attr(ctx, static_cast<uint32_t>(bitPattern.size())));

    nceOp.addDPUTask(rewriter, startAttr, endAttr, padsBeginAttr, padsEndAttr, VPUIP::MPEMode::VECTOR_FP16);

    // CMX -> DDR
    auto cmx_nhwc_mem_ref = nceOp.output().getType().dyn_cast<mlir::MemRefType>();
    auto ddr_nhwc_type = mlir::MemRefType::get(cmx_nhwc_mem_ref.getShape(), cmx_nhwc_mem_ref.getElementType(),
                                               cmx_nhwc_mem_ref.getAffineMaps(),
                                               origOp.output().getType().dyn_cast<mlir::MemRefType>().getMemorySpace());
    auto ddr_nhwc_alloc_op = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), ddr_nhwc_type);
    auto output_ddr_nhwc = rewriter.create<IERT::CopyOp>(origOp->getLoc(), nceOp.output(), ddr_nhwc_alloc_op);

    // NHWC -> NCHW
    auto outputReorderOp = rewriter.create<IERT::ReorderOp>(origOp->getLoc(), output_ddr_nhwc, origOp.output_buff());

    rewriter.replaceOp(origOp, outputReorderOp->getResults());

    return mlir::success();
}

void ConvertToNCEOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto resources = IERT::RunTimeResourcesOp::getFromModule(module);
    if (!resources) {
        _log.error("Could not retrieve IERT.RunTimeResources");
        signalPassFailure();
    }

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.insert<ConvRewrite>(&ctx, _log, resources);
    patterns.insert<MaxPoolRewrite>(&ctx, _log, resources);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createConvertToNCEOpsPass(Logger log) {
    return std::make_unique<ConvertToNCEOpsPass>(log);
}

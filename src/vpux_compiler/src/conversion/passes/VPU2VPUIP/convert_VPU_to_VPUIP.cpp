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

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Transforms/Bufferize.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

int32_t getWeightPtrStep(mlir::Value weights, mlir::Value activation_window) {
    if (weights == nullptr) {
        return 0;
    }

    const auto filterShape = getShape(weights);

    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    if (activation_window != nullptr) {
        // Depthwise convolution case.
        // Weights table contains both activation window and weights.
        // Check that weights have expected alignment.
        // Other than that, weight step is the same for both z-major (OYXI) and depthwise convolutions.
        const auto origFilterType = weights.getType().cast<vpux::NDTypeInterface>();
        const auto depthwiseConvAlignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());
        const auto weightsElementCount = IC * KY * KX;
        VPUX_THROW_UNLESS(weightsElementCount % depthwiseConvAlignment == 0,
                          "Depthwise convolution weights size must be a multiple of {0}, got {1}",
                          depthwiseConvAlignment, weightsElementCount);
    }

    const Byte eltSize = getElemTypeSize(weights.getType());
    return checked_cast<int32_t>(IC * KY * KX * eltSize.count());
}

void addPPETask(mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp& nceOp, VPU::PPETaskAttr ppeAttr) {
    const auto multList =
            ppeAttr.quant_mult() != nullptr
                    ? builder.getI32ArrayAttr(makeArrayRef(parseIntArrayAttr<int32_t>(ppeAttr.quant_mult())))
                    : nullptr;
    const auto shiftList =
            ppeAttr.quant_shift() != nullptr
                    ? builder.getI32ArrayAttr(makeArrayRef(parseIntArrayAttr<int32_t>(ppeAttr.quant_shift())))
                    : nullptr;
    nceOp.addPPETask(builder, ppeAttr.mode(), ppeAttr.clamp_low(), ppeAttr.clamp_high(), ppeAttr.lrelu_mult(),
                     ppeAttr.lrelu_shift(), multList, shiftList, ppeAttr.quant_post_shift());
}

mlir::Value createWeightsTableTensor(mlir::PatternRewriter& rewriter, mlir::Location loc, int64_t OC,
                                     mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                     mlir::Value activationWindow, Const::ContentAttr bias,
                                     vpux::VPU::PPETaskAttr ppeTaskAttr, VPU::ArchKind arch) {
    SmallVector<int64_t> weightTableShape{OC, 1, 1, VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};

    const auto dataType = mlir::MemRefType::get(weightTableShape, getSInt32Type(rewriter.getContext()));
    // Actual weight and sparsity pointers are not known at this stage, so the constant
    // is filled only with offsets from the base pointers. Once the memory scheduler
    // allocates the memory and the pointers are known the transformation is added to the
    // constant. Finally the transformation shall add the base pointers to the offsets.
    const auto weightPtrOffset = 0;
    const auto sparsityPtrOffset = 0;
    const auto weightPtrStep = getWeightPtrStep(weights, activationWindow);

    const auto opInElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto opOutElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto opWeightsElemType = weights ? weights.getType().cast<vpux::NDTypeInterface>().getElementType() : nullptr;
    const auto weightsTable =
            VPU::NCESparsity::getWeightsTable(opInElemType, opOutElemType, weightPtrOffset, weightPtrStep,
                                              sparsityPtrOffset, arch, OC, opWeightsElemType, bias, ppeTaskAttr);

    const auto dataStorageType = mlir::RankedTensorType::get(dataType.getShape(), getSInt32Type(rewriter.getContext()));
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, makeArrayRef(weightsTable));
    auto declareOp = rewriter.create<Const::DeclareOp>(loc, dataType, Const::ContentAttr::get(dataAttr));
    const auto dataTypeCMX = dataType.cast<vpux::NDTypeInterface>().changeMemSpace(VPU::MemoryKind::CMX_NN);

    auto dataAllocOp = rewriter.create<mlir::memref::AllocOp>(loc, dataTypeCMX.cast<mlir::MemRefType>());
    auto copyOp = rewriter.create<IERT::CopyOp>(loc, declareOp.output(), dataAllocOp);

    return copyOp.output();
}

void addDPUTasks(VPUIP::NCEClusterTaskOp nceOp, mlir::PatternRewriter& rewriter, mlir::Region& workloads) {
    for (auto dpuTaskOp : workloads.getOps<VPU::DPUWorkloadOp>()) {
        SmallVector<int64_t> ends;
        const auto offsets = parseIntArrayAttr<int64_t>(dpuTaskOp.offsets());
        const auto sizes = parseIntArrayAttr<int64_t>(dpuTaskOp.sizes());
        ends.reserve(sizes.size());

        llvm::transform(llvm::seq<size_t>(0, sizes.size()), std::back_inserter(ends), [&](size_t index) {
            return offsets[index] + sizes[index] - 1;
        });

        // as soon as we need workload_x, workload_y, workload_z coords
        const auto dpuStart = {offsets[Dims4D::Act::W.ind()], offsets[Dims4D::Act::H.ind()],
                               offsets[Dims4D::Act::C.ind()]};
        const auto dpuEnds = {ends[Dims4D::Act::W.ind()], ends[Dims4D::Act::H.ind()], ends[Dims4D::Act::C.ind()]};

        nceOp.addDPUTask(rewriter, getIntArrayAttr(rewriter, dpuStart), getIntArrayAttr(rewriter, dpuEnds),
                         dpuTaskOp.pad(), dpuTaskOp.mpe_mode());
    }
}

//
// Buffer allocation
//
mlir::Value allocateResult(mlir::Location loc, mlir::OpBuilder& builder, mlir::TypeConverter& typeConverter,
                           mlir::Value output) {
    auto origType = output.getType();
    auto memRefType = typeConverter.convertType(origType);
    auto allocOp = builder.create<mlir::memref::AllocOp>(loc, memRefType.cast<mlir::MemRefType>());
    return allocOp.memref();
}

mlir::Value createActivationWindowTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<uint8_t> fakeSparsity,
                                         int64_t numChannels) {
    const auto elemType = getUInt8Type(builder.getContext());

    SmallVector<int64_t> fakeSparsityShape{numChannels, 1, 1, static_cast<int64_t>(fakeSparsity.size()) / numChannels};

    const auto dataStorageType = mlir::RankedTensorType::get(fakeSparsityShape, elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, fakeSparsity);

    const auto dataType = mlir::MemRefType::get(fakeSparsityShape, elemType);
    auto dataConstOp = builder.create<Const::DeclareOp>(loc, dataType, Const::ContentAttr::get(dataAttr));

    const auto dataTypeCMX = dataType.cast<vpux::NDTypeInterface>().changeMemSpace(VPU::MemoryKind::CMX_NN);

    auto dataAllocOp = builder.create<mlir::memref::AllocOp>(loc, dataTypeCMX.cast<mlir::MemRefType>());
    auto copyOp = builder.create<IERT::CopyOp>(loc, dataConstOp.output(), dataAllocOp);

    return copyOp.output();
}

//
// ConvRewriter
//

class ConvRewriter final : public mlir::OpConversionPattern<VPU::NCEConvolutionOp> {
public:
    ConvRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log, VPU::ArchKind arch)
            : mlir::OpConversionPattern<VPU::NCEConvolutionOp>(typeConverter, ctx), _log(log), _arch(arch) {
        setDebugName("ConvRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::ArchKind _arch;
};

mlir::LogicalResult ConvRewriter::matchAndRewrite(VPU::NCEConvolutionOp origOp, OpAdaptor newArgs,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    //
    // Buffer allocation
    //

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto outputBuffer = allocateResult(origOp.getLoc(), rewriter, *typeConverter, origOp.output());

    //
    // Get dimensions
    //

    const Shape filterShape = origOp.rawFilterShapeAttr() != nullptr
                                      ? Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()))
                                      : getShape(newArgs.filter()).toValues();

    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto inOrder = DimsOrder::fromValue(newArgs.input());
    const auto isCMajor = inOrder == DimsOrder::NCHW;

    //
    // Generate activation window
    //

    mlir::IntegerAttr actWindowChanLen;
    mlir::Value activationWindow;

    if (isCMajor) {
        const auto origInputType = origOp.input().getType().cast<vpux::NDTypeInterface>();

        const auto kernelSize = Shape{KY, KX};
        const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));

        const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::CM_CONV, kernelSize,
                                                                        kernelStrides[Dims4D::Strides::X],
                                                                        origInputType.getElementType(), IC);

        const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(VPU::NCESparsity::Mode::CM_CONV, kernelSize,
                                                                    kernelStrides[Dims4D::Strides::X],
                                                                    origInputType.getElementType(), IC, OC);

        actWindowChanLen = getIntAttr(getContext(), bitPatternSize);
        activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity, OC);
    }

    //
    // Prepare output buffer for DPU
    //

    auto weightsTable =
            createWeightsTableTensor(rewriter, origOp->getLoc(), OC, newArgs.input(), outputBuffer, newArgs.filter(),
                                     activationWindow, origOp.biasAttr(), origOp.ppeAttr(), _arch);

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(getContext(), makeArrayRef({KY, KX}));
    const auto taskType = isCMajor ? VPUIP::NCETaskType::CMCONV : VPUIP::NCETaskType::CONV;

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(origOp->getLoc(), newArgs.input(), newArgs.filter(),
                                                          weightsTable, activationWindow,
                                                          /*parent_input=*/newArgs.input(),
                                                          /*parent_output=*/outputBuffer,
                                                          /*output_buff=*/outputBuffer, taskType, kernelSizeAttr,
                                                          origOp.strides(), origOp.padAttr(), actWindowChanLen);

    addDPUTasks(nceOp, rewriter, origOp.workloads());

    if (origOp.ppe().hasValue()) {
        addPPETask(rewriter, nceOp, origOp.ppeAttr());
    }

    rewriter.replaceOp(origOp, nceOp.output());

    return mlir::success();
}

//
// MaxPoolRewriter
//

class MaxPoolRewriter final : public mlir::OpConversionPattern<VPU::NCEMaxPoolOp> {
public:
    MaxPoolRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log, VPU::ArchKind arch)
            : mlir::OpConversionPattern<VPU::NCEMaxPoolOp>(typeConverter, ctx), _log(log), _arch(arch) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEMaxPoolOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::ArchKind _arch;
};

mlir::LogicalResult MaxPoolRewriter::matchAndRewrite(VPU::NCEMaxPoolOp origOp, OpAdaptor newArgs,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    //
    // Buffer allocation
    //

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto outputBuffer = allocateResult(origOp.getLoc(), rewriter, *typeConverter, origOp.output());

    //
    // Get dimensions
    //

    const auto origInputType = newArgs.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = origInputType.getShape();

    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(origOp.kernel_size()));
    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));

    const auto bitPatternSize =
            VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::POOL, kernelSize,
                                                kernelStrides[Dims4D::Strides::X], origInputType.getElementType(), IC);

    //
    // Generate activation window
    //

    const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(VPU::NCESparsity::Mode::POOL, kernelSize,
                                                                kernelStrides[Dims4D::Strides::X],
                                                                origInputType.getElementType(), IC, IC);
    const auto activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity, IC);

    //
    // Prepare output buffer for DPU
    //

    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), IC, newArgs.input(), outputBuffer, nullptr,
                                                 activationWindow, nullptr, origOp.ppeAttr(), _arch);

    //
    // Create NCE per-cluster Operation
    //

    const auto activation_window_channel_length = getIntAttr(getContext(), static_cast<uint32_t>(bitPatternSize));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), newArgs.input(), /*weights=*/nullptr, weightsTable, activationWindow,
            /*parent_input=*/newArgs.input(),
            /*parent_output=*/outputBuffer,
            /*output_buff=*/outputBuffer, VPUIP::NCETaskType::MAXPOOL, origOp.kernel_size(), origOp.strides(),
            origOp.pad(), activation_window_channel_length);

    addDPUTasks(nceOp, rewriter, origOp.workloads());

    if (origOp.ppe().hasValue()) {
        addPPETask(rewriter, nceOp, origOp.ppeAttr());
    }

    rewriter.replaceOp(origOp, nceOp.output());

    return mlir::success();
}

//
// DepthwiseConvRewriter
//

class DepthwiseConvRewriter final : public mlir::OpConversionPattern<VPU::NCEDepthConvolutionOp> {
public:
    DepthwiseConvRewriter(mlir::TypeConverter& converter, mlir::MLIRContext* ctx, Logger log, VPU::ArchKind arch)
            : mlir::OpConversionPattern<VPU::NCEDepthConvolutionOp>(converter, ctx), _log(log), _arch(arch) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEDepthConvolutionOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::ArchKind _arch;
};

mlir::LogicalResult DepthwiseConvRewriter::matchAndRewrite(VPU::NCEDepthConvolutionOp origOp, OpAdaptor newArgs,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    //
    // Buffer allocation
    //

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    //
    // Get dimensions
    //

    const Shape filterShape = origOp.rawFilterShapeAttr() != nullptr
                                      ? Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()))
                                      : getShape(newArgs.filter()).toValues();

    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    //
    // Generate activation window
    //

    const auto origInputType = newArgs.input().getType().cast<vpux::NDTypeInterface>();

    const auto origInputShape = origInputType.getShape();
    const auto IC = origInputShape[Dims4D::Act::C];

    const auto kernelSize = Shape{KY, KX};
    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));
    const auto bitPatternSize =
            VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::DW_CONV, kernelSize,
                                                kernelStrides[Dims4D::Strides::X], origInputType.getElementType(), IC);
    const auto actWindowChanLen = getIntAttr(getContext(), bitPatternSize);

    const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(VPU::NCESparsity::Mode::DW_CONV, kernelSize,
                                                                kernelStrides[Dims4D::Strides::X],
                                                                origInputType.getElementType(), IC, OC);
    const auto activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity, OC);

    //
    // Prepare output buffer for DPU
    //
    const auto outputBuffer = allocateResult(origOp.getLoc(), rewriter, *typeConverter, origOp.output());

    auto weightsTable =
            createWeightsTableTensor(rewriter, origOp->getLoc(), OC, newArgs.input(), outputBuffer, newArgs.filter(),
                                     activationWindow, origOp.biasAttr(), origOp.ppeAttr(), _arch);

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(getContext(), makeArrayRef({KY, KX}));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), newArgs.input(), newArgs.filter(), weightsTable, activationWindow,
            /*parent_input=*/newArgs.input(),
            /*parent_output=*/outputBuffer,
            /*output_buff=*/outputBuffer, VPUIP::NCETaskType::DWCONV, kernelSizeAttr, origOp.strides(), origOp.pad(),
            actWindowChanLen);

    addDPUTasks(nceOp, rewriter, origOp.workloads());

    if (origOp.ppe().hasValue()) {
        addPPETask(rewriter, nceOp, origOp.ppeAttr());
    }

    rewriter.replaceOp(origOp, nceOp.output());

    return mlir::success();
}

//
// EltwiseRewriter
//

class EltwiseRewriter final : public mlir::OpConversionPattern<VPU::NCEEltwiseOp> {
public:
    EltwiseRewriter(mlir::TypeConverter& converter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::NCEEltwiseOp>(converter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEEltwiseOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::ArchKind _arch;
};

mlir::LogicalResult EltwiseRewriter::matchAndRewrite(VPU::NCEEltwiseOp origOp, OpAdaptor newArgs,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    //
    // Buffer allocation
    //

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffer = allocateResult(origOp.getLoc(), rewriter, *typeConverter, origOp.output());

    //
    // Create NCE per-cluster Operation
    //

    const auto activation_window_channel_length = getIntAttr(this->getContext(), static_cast<int32_t>(0));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(origOp->getLoc(), newArgs.input1(), newArgs.input2(),
                                                          /*weightsTable=*/nullptr,
                                                          /*activation_window=*/nullptr,
                                                          /*parent_input=*/newArgs.input1(),
                                                          /*parent_output=*/outputBuffer,
                                                          /*output_buff=*/outputBuffer, VPUIP::NCETaskType::ELTWISE,
                                                          /*kernel_size=*/nullptr,
                                                          /*kernel_strides=*/nullptr,
                                                          /*kernel_padding=*/nullptr, activation_window_channel_length);

    //
    // Create DPU sub-task
    //

    addDPUTasks(nceOp, rewriter, origOp.workloads());

    VPUX_THROW_UNLESS(origOp.ppe().hasValue(), "Eltwise operation should always have PPE info");

    addPPETask(rewriter, nceOp, origOp.ppeAttr());

    rewriter.replaceOp(origOp, nceOp.output());

    return mlir::success();
}

//
// ConvertVPUToVPUIPPass
//

class ConvertVPUToVPUIPPass final : public ConvertVPUToVPUIPBase<ConvertVPUToVPUIPPass> {
public:
    explicit ConvertVPUToVPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertVPUToVPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    vpux::BufferizeTypeConverter typeConverter;

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalDialect<Const::ConstDialect>(isLegalOp);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<IERT::IERTDialect>();
    target.addIllegalDialect<VPU::VPUDialect>();
    target.addLegalOp<mlir::memref::AllocOp>();
    vpux::populateBufferizeMaterializationLegality(target);

    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvRewriter>(typeConverter, &ctx, _log, arch);
    patterns.add<DepthwiseConvRewriter>(typeConverter, &ctx, _log, arch);
    patterns.add<MaxPoolRewriter>(typeConverter, &ctx, _log, arch);
    patterns.add<EltwiseRewriter>(typeConverter, &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertVPUToVPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUToVPUIPPass(Logger log) {
    return std::make_unique<ConvertVPUToVPUIPPass>(log);
}

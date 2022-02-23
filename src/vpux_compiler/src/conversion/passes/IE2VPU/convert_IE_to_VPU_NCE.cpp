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

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

mlir::Value copyToCMX(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val) {
    if (val == nullptr) {
        return nullptr;
    }
    const auto memSpace = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
    return builder.createOrFold<IE::CopyOp>(loc, val, memSpace);
}

mlir::Value copyBack(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val, mlir::Type origType) {
    const auto memSpace = IE::getMemorySpace(origType.cast<mlir::RankedTensorType>());
    return builder.createOrFold<IE::CopyOp>(loc, val, memSpace);
}

mlir::Value createActivationWindowTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<uint8_t> fakeSparsity,
                                         int64_t numChannels) {
    const auto elemType = getUInt8Type(builder.getContext());
    SmallVector<int64_t> fakeSparsityShape{numChannels, 1, 1, static_cast<int64_t>(fakeSparsity.size()) / numChannels};

    const auto dataStorageType = mlir::RankedTensorType::get(fakeSparsityShape, elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, fakeSparsity);

    auto dataConstOp = builder.create<Const::DeclareOp>(loc, dataStorageType, Const::ContentAttr::get(dataAttr));
    return dataConstOp.output();
}

int32_t getWeightPtrStep(::mlir::Value weights, ::mlir::Value activation_window) {
    if (weights == nullptr) {
        return 0;
    }

    const auto filterShape = getShape(weights);

    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    if (activation_window != nullptr) {
        // Channel major and depthwise convolution case.
        // Weights table contains both activation window and weights.
        // Check that weights have expected alignment.
        // Other than that, weight step is the same for both z-major (OYXI) and depthwise convolutions.
        const auto origFilterType = weights.getType().cast<vpux::NDTypeInterface>();
        const auto convAlignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());
        const auto weightsElementCount = IC * KY * KX;
        VPUX_THROW_UNLESS(weightsElementCount % convAlignment == 0,
                          "Channel Major and Depthwise convolution weights size must be a multiple of {0}, got {1}",
                          convAlignment, weightsElementCount);
    }

    const Byte eltSize = getElemTypeSize(weights.getType());
    return checked_cast<int32_t>(IC * KY * KX * eltSize.count());
}

VPU::PPETaskAttr getPPETaskAttr(mlir::Location loc, mlir::Value opInput, mlir::Value opOutput,
                                vpux::IE::PostOp postOpAttr, mlir::MLIRContext* context, VPU::ArchKind arch) {
    const auto inElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();

    const auto postOpParams = VPU::parsePostOp(postOpAttr, inElemType, outElemType, arch, loc);

    VPU::PPETaskAttr ppeTaskAttr;
    if (postOpParams.hasValue()) {
        ppeTaskAttr = VPU::getPPEAttr(postOpParams.getValue(), context);
    }

    return ppeTaskAttr;
}

std::vector<int32_t> createWeightsTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                            mlir::Value activationWindow, Const::ContentAttr bias, int64_t OC,
                                            vpux::VPU::PPETaskAttr ppeTaskAttr, VPU::ArchKind _arch) {
    const auto weightPtrOffset = 0;
    const auto sparsityPtrOffset = 0;
    const auto weightPtrStep = getWeightPtrStep(weights, activationWindow);

    const auto inElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto weightsElemType = weights ? weights.getType().cast<vpux::NDTypeInterface>().getElementType() : nullptr;

    return VPU::NCESparsity::getWeightsTable(inElemType, outElemType, weightPtrOffset, weightPtrStep, sparsityPtrOffset,
                                             _arch, OC, weightsElemType, bias, ppeTaskAttr);
}

mlir::Value createWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<int32_t> weightsTable,
                                     int64_t OC) {
    const auto elemType = getSInt32Type(builder.getContext());
    SmallVector<int64_t> weightTableShape{OC, 1, 1, VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};

    const auto dataStorageType = mlir::RankedTensorType::get(weightTableShape, elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, weightsTable);

    auto dataConstOp = builder.create<Const::DeclareOp>(loc, dataStorageType, Const::ContentAttr::get(dataAttr));
    return dataConstOp.output();
}

//
// ConvToNCE
//

class ConvToNCE final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvToNCE(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _arch(arch), _log(log) {
        setDebugName("ConvToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult ConvToNCE::matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    const auto inOrder = DimsOrder::fromValue(origOp.input());
    if (inOrder != DimsOrder::NCHW && inOrder != DimsOrder::NHWC) {
        return matchFailed(_log, rewriter, origOp, "Operation at '{0}' has unsupported input layout '{1}'",
                           origOp->getLoc(), inOrder);
    }

    const auto isCMajor = inOrder == DimsOrder::NCHW;

    if (!VPU::NCEConvolutionOp::isSupported(origOp, logCb)) {
        return mlir::failure();
    }

    Const::ContentAttr bias;
    if (origOp.bias() != nullptr) {
        auto biasConstOp = origOp.bias().getDefiningOp<Const::DeclareOp>();
        if (biasConstOp == nullptr) {
            return matchFailed(_log, rewriter, origOp, "[{0}] '{1}' at '{2}' has non constant biases", getDebugName(),
                               origOp->getName(), origOp->getLoc());
        }

        bias = biasConstOp.contentAttr();
    }

    auto filter = origOp.filter();
    const auto filterShape = getShape(filter);
    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    // Generate activation window
    mlir::IntegerAttr activationWindowChannelLength;
    mlir::Value activationWindow = nullptr;

    if (isCMajor) {
        const auto kernelSize = Shape{KY, KX};
        const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));
        const auto origInputType = origOp.input().getType().cast<vpux::NDTypeInterface>();

        const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::CM_CONV, kernelSize,
                                                                        kernelStrides[Dims4D::Strides::X],
                                                                        origInputType.getElementType(), IC);

        const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(VPU::NCESparsity::Mode::CM_CONV, kernelSize,
                                                                    kernelStrides[Dims4D::Strides::X],
                                                                    origInputType.getElementType(), IC, OC);

        activationWindowChannelLength = getIntAttr(getContext(), bitPatternSize);
        activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity, OC);

        filter = VPU::alignChannelMajorWeightsTensor(rewriter, origOp->getLoc(), filter);
    }

    // Generate weights table
    auto ppeTaskAttr = getPPETaskAttr(origOp.getLoc(), origOp.input(), origOp.output(), origOp.post_opAttr(),
                                      origOp.getContext(), _arch);
    auto weightsTableVec = createWeightsTableData(origOp.input(), origOp.output(), filter, activationWindow, bias, OC,
                                                  ppeTaskAttr, _arch);
    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec, OC);

    const auto inputCMX = copyToCMX(rewriter, appendLoc(origOp->getLoc(), "input-CMX"), origOp.input());
    const auto filterCMX = copyToCMX(rewriter, appendLoc(origOp->getLoc(), "filter-CMX"), filter);
    const auto weightsTableCMX = copyToCMX(rewriter, appendLoc(origOp->getLoc(), "weights-table-CMX"), weightsTable);
    const auto activationWindowCMX =
            copyToCMX(rewriter, appendLoc(origOp->getLoc(), "activation-window-CMX"), activationWindow);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.pads_begin(), origOp.pads_end()));
    const auto outputType = origOp.getType().cast<vpux::NDTypeInterface>();
    const auto newOutType = outputType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    const auto rawFilterShape = getIntArrayAttr(rewriter, getShape(origOp.filter()));

    auto nceOp = rewriter.create<VPU::NCEConvolutionOp>(origOp->getLoc(), newOutType, inputCMX, filterCMX,
                                                        weightsTableCMX, activationWindowCMX, bias,
                                                        origOp.stridesAttr(), padAttr, /*post_op=*/nullptr, ppeTaskAttr,
                                                        rawFilterShape, activationWindowChannelLength);

    const auto newOutput =
            copyBack(rewriter, appendLoc(origOp->getLoc(), "output-DDR"), nceOp.output(), origOp.getType());

    rewriter.replaceOp(origOp, newOutput);
    return mlir::success();
}

//
// DepthConvToNCE
//

class DepthConvToNCE final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    DepthConvToNCE(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult DepthConvToNCE::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    if (!VPU::NCEDepthConvolutionOp::isSupported(origOp, logCb)) {
        return mlir::failure();
    }

    // Get dimensions
    const auto alignedFilter = VPU::alignDepthWiseWeightsTensor(rewriter, origOp.getLoc(), origOp.filter());
    const auto filterShape = getShape(alignedFilter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    // Generate activation window
    const auto origInputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto origInputShape = origInputType.getShape();
    const auto IC = origInputShape[Dims4D::Act::C];

    const auto kernelSize = Shape{KY, KX};
    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));
    const auto bitPatternSize =
            VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::DW_CONV, kernelSize,
                                                kernelStrides[Dims4D::Strides::X], origInputType.getElementType(), IC);
    const auto activationWindowChannelLength = getIntAttr(getContext(), bitPatternSize);

    const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(VPU::NCESparsity::Mode::DW_CONV, kernelSize,
                                                                kernelStrides[Dims4D::Strides::X],
                                                                origInputType.getElementType(), IC, OC);
    const auto activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity, OC);

    Const::ContentAttr bias;
    if (origOp.bias() != nullptr) {
        auto biasConstOp = origOp.bias().getDefiningOp<Const::DeclareOp>();
        if (biasConstOp == nullptr) {
            return matchFailed(_log, rewriter, origOp, "[{0}] '{1}' at '{2}' has non constant biases", getDebugName(),
                               origOp->getName(), origOp->getLoc());
        }

        bias = biasConstOp.contentAttr();
    }

    // Generate weights table
    auto ppeTaskAttr = getPPETaskAttr(origOp.getLoc(), origOp.input(), origOp.output(), origOp.post_opAttr(),
                                      origOp.getContext(), _arch);
    auto weightsTableVec = createWeightsTableData(origOp.input(), origOp.output(), alignedFilter, activationWindow,
                                                  bias, OC, ppeTaskAttr, _arch);
    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec, OC);

    const auto inputCMX = copyToCMX(rewriter, appendLoc(origOp->getLoc(), "input-CMX"), origOp.input());
    const auto filterCMX = copyToCMX(rewriter, appendLoc(origOp->getLoc(), "filter-CMX"), alignedFilter);
    const auto weightsTableCMX = copyToCMX(rewriter, appendLoc(origOp->getLoc(), "weights-table-CMX"), weightsTable);
    const auto activationWindowCMX =
            copyToCMX(rewriter, appendLoc(origOp->getLoc(), "activation-window-CMX"), activationWindow);
    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.pads_begin(), origOp.pads_end()));
    const auto outputType = origOp.getType().cast<vpux::NDTypeInterface>();
    const auto newOutType = outputType.changeMemSpace(VPU::MemoryKind::CMX_NN);
    const auto rawFilterShape = getIntArrayAttr(rewriter, getShape(origOp.filter()));

    auto nceOp = rewriter.create<VPU::NCEDepthConvolutionOp>(
            origOp->getLoc(), newOutType, inputCMX, filterCMX, weightsTableCMX, activationWindowCMX, bias,
            origOp.stridesAttr(), padAttr, /*post_op=*/nullptr, ppeTaskAttr, rawFilterShape,
            activationWindowChannelLength);

    const auto newOutput =
            copyBack(rewriter, appendLoc(origOp->getLoc(), "output-DDR"), nceOp.output(), origOp.getType());

    rewriter.replaceOp(origOp, newOutput);
    return mlir::success();
}

//
// MaxPoolToNCE
//

class MaxPoolToNCE final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolToNCE(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult MaxPoolToNCE::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    if (!VPU::NCEMaxPoolOp::isSupported(origOp, logCb)) {
        return mlir::failure();
    }

    // Get dimensions
    const auto origInputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = origInputType.getShape();

    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(origOp.kernel_size()));
    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));

    const auto bitPatternSize =
            VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::POOL, kernelSize,
                                                kernelStrides[Dims4D::Strides::X], origInputType.getElementType(), IC);

    // Generate activation window
    const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(VPU::NCESparsity::Mode::POOL, kernelSize,
                                                                kernelStrides[Dims4D::Strides::X],
                                                                origInputType.getElementType(), IC, IC);
    const auto activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity, IC);
    const auto activationWindowChannelLength = getIntAttr(getContext(), static_cast<uint32_t>(bitPatternSize));

    // Generate weights table
    auto ppeTaskAttr = getPPETaskAttr(origOp.getLoc(), origOp.input(), origOp.output(), origOp.post_opAttr(),
                                      origOp.getContext(), _arch);
    auto weightsTableVec = createWeightsTableData(origOp.input(), origOp.output(), nullptr, activationWindow, nullptr,
                                                  IC, ppeTaskAttr, _arch);
    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec, IC);

    const auto inputCMX = copyToCMX(rewriter, appendLoc(origOp->getLoc(), "input-CMX"), origOp.input());
    auto weightsTableCMX = copyToCMX(rewriter, appendLoc(origOp->getLoc(), "weights-table-CMX"), weightsTable);
    const auto activationWindowCMX =
            copyToCMX(rewriter, appendLoc(origOp->getLoc(), "activation-window-CMX"), activationWindow);
    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.pads_begin(), origOp.pads_end()));
    const auto outputType = origOp.getType().cast<vpux::NDTypeInterface>();
    const auto newOutType = outputType.changeMemSpace(VPU::MemoryKind::CMX_NN);

    auto nceOp = rewriter.create<VPU::NCEMaxPoolOp>(
            origOp->getLoc(), newOutType, inputCMX, weightsTableCMX, activationWindowCMX, origOp.kernel_sizeAttr(),
            origOp.stridesAttr(), padAttr, /*post_op=*/nullptr, ppeTaskAttr, activationWindowChannelLength);

    const auto newOutput =
            copyBack(rewriter, appendLoc(origOp->getLoc(), "output-DDR"), nceOp.output(), origOp.getType());

    rewriter.replaceOp(origOp, newOutput);
    return mlir::success();
}

//
// EltwiseToNCE
//

template <class ConcreteOp>
class EltwiseToNCE final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    EltwiseToNCE<ConcreteOp>(mlir::MLIRContext* ctx, VPU::EltwiseType opType, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _opType(opType), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::EltwiseType _opType;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult EltwiseToNCE<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", this->getDebugName(), msg.str());
    };

    const bool allowDifferentScales = _opType == VPU::EltwiseType::MULTIPLY;
    const bool allowDifferentZp = true;

    if (!VPU::NCEEltwiseOp::isSupported(origOp, allowDifferentScales, allowDifferentZp, logCb)) {
        return mlir::failure();
    }

    const auto inputCMX1 = copyToCMX(rewriter, appendLoc(origOp->getLoc(), "input1-CMX"), origOp.input1());
    const auto inputCMX2 = origOp.input1() == origOp.input2()
                                   ? inputCMX1
                                   : copyToCMX(rewriter, appendLoc(origOp->getLoc(), "input2-CMX"), origOp.input2());

    const auto outputType = origOp.getType().template cast<vpux::NDTypeInterface>();
    const auto newOutType = outputType.changeMemSpace(VPU::MemoryKind::CMX_NN);

    auto nceOp = rewriter.create<VPU::NCEEltwiseOp>(origOp->getLoc(), newOutType, inputCMX1, inputCMX2,
                                                    VPU::EltwiseTypeAttr::get(this->getContext(), _opType),
                                                    origOp.post_opAttr(),
                                                    /*ppe=*/nullptr);

    const auto newOutput =
            copyBack(rewriter, appendLoc(origOp->getLoc(), "output-DDR"), nceOp.output(), origOp.getType());

    rewriter.replaceOp(origOp, newOutput);
    return mlir::success();
}

//
// ConvertIEToVPUNCEPass
//

class ConvertIEToVPUNCEPass final : public ConvertIEToVPUNCEBase<ConvertIEToVPUNCEPass> {
public:
    explicit ConvertIEToVPUNCEPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertIEToVPUNCEPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.add<ConvToNCE>(&ctx, arch, _log);
    patterns.add<DepthConvToNCE>(&ctx, arch, _log);
    patterns.add<MaxPoolToNCE>(&ctx, arch, _log);
    patterns.add<EltwiseToNCE<IE::AddOp>>(&ctx, VPU::EltwiseType::ADD, _log);
    patterns.add<EltwiseToNCE<IE::MultiplyOp>>(&ctx, VPU::EltwiseType::MULTIPLY, _log);
    patterns.add<EltwiseToNCE<IE::SubtractOp>>(&ctx, VPU::EltwiseType::SUBTRACT, _log);
    patterns.add<EltwiseToNCE<IE::AndOp>>(&ctx, VPU::EltwiseType::AND, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertIEToVPUNCENCEPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertIEToVPUNCEPass(Logger log) {
    return std::make_unique<ConvertIEToVPUNCEPass>(log);
}

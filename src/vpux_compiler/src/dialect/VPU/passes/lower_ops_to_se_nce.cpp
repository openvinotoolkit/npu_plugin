//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/IE/loop.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// InterpolateToNCE
//

class InterpolateToNCE final : public mlir::OpRewritePattern<VPU::InterpolateOp> {
public:
    InterpolateToNCE(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::InterpolateOp>(ctx), _arch(arch), _log(log) {
        setDebugName("InterpolateToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::NCEInterpolateModeAttr getModeAttr(IE::InterpolateModeAttr origModeAttr) const;
    int64_t getSESize(int64_t channels) const;

    mlir::Value createSparseInput(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter,
                                  VPU::NCEInterpolateModeAttr modeAttr, ArrayRef<double> scales) const;
    mlir::Value createWeightsConstant(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter,
                                      VPU::NCEInterpolateModeAttr modeAttr, ArrayRef<double> scales) const;

    VPU::ArchKind _arch;
    Logger _log;
};

VPU::NCEInterpolateModeAttr InterpolateToNCE::getModeAttr(IE::InterpolateModeAttr origModeAttr) const {
    if (origModeAttr == nullptr) {
        return nullptr;
    }

    auto ctx = origModeAttr.getContext();
    auto origMode = origModeAttr.getValue();
    switch (origMode) {
    case IE::InterpolateMode::NEAREST:
        return VPU::NCEInterpolateModeAttr::get(ctx, VPU::NCEInterpolateMode::NEAREST);
    case IE::InterpolateMode::LINEAR:
    case IE::InterpolateMode::LINEAR_ONNX:
        return VPU::NCEInterpolateModeAttr::get(ctx, VPU::NCEInterpolateMode::BILINEAR);
    default:
        return nullptr;
    }
}

// Get the largest storage element size that is compatible with the given number of channels
// The channels must be a multiple of 16 and the maximum value supported is 8192; the same
// restriction applies to the storage element size
int64_t InterpolateToNCE::getSESize(int64_t channels) const {
    for (int64_t seSize = VPU::NCEInvariant::VPU_DIMENSION_LIMIT; seSize >= 16; seSize /= 2) {
        if (channels % seSize == 0) {
            return seSize;
        }
    }
    VPUX_THROW("Failed to find se_size for '{0}' channels", channels);
}

// Creates a sparse input whose sparsity map and storage element table have the following shapes:
//   NEAREST: ScaleH x ScaleW
//   BILINEAR [ScaleH + (ScaleH-1)] x [ScaleW + (ScaleW-1)]
// The sparsity map constant has all bits set to 1.
// The storage element table operation and the resulting sparse tensor have a SEInterpolateAttr set
// which defines the relationship between the input data and sparsity metadata.
mlir::Value InterpolateToNCE::createSparseInput(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter,
                                                VPU::NCEInterpolateModeAttr modeAttr, ArrayRef<double> scales) const {
    auto ctx = origOp.getContext();
    auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();
    auto inputDimsOrder = inputType.getDimsOrder();

    // Create the SEInterpolateAttr
    auto coordModeAttr = origOp.attr().getCoordMode();
    VPUX_THROW_WHEN(coordModeAttr == nullptr, "Missing coordinate transformation mode");
    IE::InterpolateNearestModeAttr nearestModeAttr = nullptr;
    if (modeAttr != nullptr && modeAttr.getValue() == VPU::NCEInterpolateMode::NEAREST) {
        nearestModeAttr = origOp.attr().getNearestMode();
        VPUX_THROW_WHEN(nearestModeAttr == nullptr, "Missing nearest mode");
    }
    auto scalesAttr = getFPArrayAttr(ctx, scales);
    auto initialInputShapeAttr = getIntArrayAttr(ctx, inputShape.raw());
    auto initialOutputShapeAttr = getIntArrayAttr(ctx, outputShape.raw());
    auto seInterpolateAttr = VPU::SEInterpolateAttr::get(ctx, modeAttr, coordModeAttr, scalesAttr, nearestModeAttr,
                                                         /*offsets=*/nullptr, /*sizes=*/nullptr, initialInputShapeAttr,
                                                         initialOutputShapeAttr);
    auto seAttr = seInterpolateAttr.cast<VPU::SEAttr>();

    // Create the StorageElementTable operation
    const int64_t seSize = getSESize(inputShape[Dims4D::Act::C]);
    const int64_t seDepth = inputShape[Dims4D::Act::C] / seSize;
    auto seTableOp = rewriter.create<VPU::StorageElementTableOp>(origOp->getLoc(), inputShape.raw(),
                                                                 inputType.getElementType(), seSize, seDepth, seAttr);

    // Create the sparsity map constant
    auto smShape = to_small_vector(seTableOp.getType().cast<vpux::NDTypeInterface>().getShape());
    smShape[Dims4D::Act::C.ind()] = seSize * seDepth;
    auto smContentElemType = mlir::IntegerType::get(ctx, 8);
    auto smContentType = mlir::RankedTensorType::get(smShape, smContentElemType);
    auto smElemCount =
            std::accumulate(smShape.begin(), smShape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    SmallVector<uint8_t> smContent(smElemCount, 1);
    const auto baseAttr = mlir::DenseElementsAttr::get(smContentType, makeArrayRef(smContent));
    auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);
    auto smElemType = mlir::IntegerType::get(ctx, 1);
    auto smType = mlir::RankedTensorType::get(smShape, smElemType, tensorAttr);
    auto contentAttr = Const::ContentAttr::get(baseAttr).reorder(inputDimsOrder).convertElemType(smElemType);
    auto smConstOp = rewriter.create<Const::DeclareOp>(origOp.getLoc(), smType, contentAttr);

    auto groupOp = rewriter.create<VPU::GroupSparseTensorOp>(origOp->getLoc(), origOp.input(), smConstOp.getOutput(),
                                                             seTableOp.output(), seAttr);
    return groupOp.output();
}

// Creates the weights constant so that the NCEConvolution operation simulates the behavior of a depthwise convolution.
// The kernels have the following configuration, where one single input channel will be populated for each kernel:
//   NEAREST: 1x1 kernel with value 1 (i.e. pass-through convolution)
//   BILINEAR: ScaleH x ScaleW kernel with value 1 / (ScaleH * ScaleW)
mlir::Value InterpolateToNCE::createWeightsConstant(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter,
                                                    VPU::NCEInterpolateModeAttr modeAttr,
                                                    ArrayRef<double> scales) const {
    auto ctx = origOp.getContext();

    auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto channels = inputType.getShape()[Dims4D::Act::C];

    Shape shape({});
    const auto mode = modeAttr.getValue();
    if (mode == VPU::NCEInterpolateMode::NEAREST) {
        shape = Shape({channels, channels, 1, 1});
    } else if (mode == VPU::NCEInterpolateMode::BILINEAR) {
        const auto scaleH = checked_cast<int64_t>(scales[Dims4D::Act::H.ind()]);
        const auto scaleW = checked_cast<int64_t>(scales[Dims4D::Act::W.ind()]);
        shape = Shape({channels, channels, scaleH, scaleW});
    } else {
        VPUX_THROW("Unsupported mode: {0}", mode);
    }

    auto weightsValue = (mode == VPU::NCEInterpolateMode::BILINEAR)
                                ? (1.0f / checked_cast<float>(shape[Dims4D::Filter::KY] * shape[Dims4D::Filter::KX]))
                                : 1.0f;

    mlir::Type elemType = mlir::Float16Type::get(ctx);
    const auto inputElemType = inputType.getElementType();
    if (auto qInputElemType = inputElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        double quantScale = 1.0;
        // Bilinear mode has each element in the weights as 1.0 / (scaleH * scaleW)
        // When the data is quantized, the weights value is set to 1 and the real value is set in the scale
        if (mode == VPU::NCEInterpolateMode::BILINEAR) {
            quantScale = static_cast<double>(weightsValue);
            weightsValue = 1.0f;
        }
        elemType = mlir::quant::UniformQuantizedType::get(
                /*flags=*/0, /*storageType=*/getUInt8Type(ctx), /*expressedType=*/mlir::Float16Type::get(ctx),
                /*scale=*/quantScale, /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);
    }
    const auto tensorAttr = vpux::getTensorAttr(ctx, DimsOrder::OYXI, nullptr);
    const auto weightsType =
            mlir::RankedTensorType::get(shape.raw(), elemType, tensorAttr).cast<vpux::NDTypeInterface>();
    const auto order = weightsType.getDimsOrder();

    const auto weightsNumElems = weightsType.getNumElements();
    SmallVector<float> content(weightsNumElems, 0.0f);
    if (mode == VPU::NCEInterpolateMode::NEAREST) {
        loop_1d(LoopExecPolicy::Parallel, channels, [&](int64_t oc) {
            content[oc * channels + oc] = weightsValue;
        });
    } else if (mode == VPU::NCEInterpolateMode::BILINEAR) {
        const auto kernelSize = shape[Dims4D::Filter::KY] * shape[Dims4D::Filter::KX];
        const auto weightSetSize = shape[Dims4D::Filter::IC] * kernelSize;
        loop_1d(LoopExecPolicy::Parallel, channels, [&](int64_t oc) {
            const auto startIC = oc * kernelSize;
            const auto endIC = startIC + kernelSize;
            for (auto ic = startIC; ic < endIC; ++ic) {
                content[oc * weightSetSize + ic] = weightsValue;
            }
        });
    }
    const auto dataStorageType = mlir::RankedTensorType::get(shape.raw(), mlir::Float32Type::get(ctx));
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, makeArrayRef(content));

    auto contentAttr = Const::ContentAttr::get(dataAttr);
    if (auto qElemType = elemType.dyn_cast<mlir::quant::QuantizedType>()) {
        contentAttr = contentAttr.convertElemType(getUInt8Type(ctx));
        contentAttr = contentAttr.quantCast(qElemType);
    } else if (elemType.isa<mlir::Float16Type>()) {
        contentAttr = contentAttr.convertElemType(mlir::Float16Type::get(ctx));
    }
    if (order != DimsOrder::fromNumDims(shape.size())) {
        contentAttr = contentAttr.reorder(order);
    }

    auto weightsConstOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), weightsType, contentAttr);
    return weightsConstOp.getOutput();
}

mlir::LogicalResult InterpolateToNCE::matchAndRewrite(VPU::InterpolateOp origOp,
                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto modeAttr = getModeAttr(origOp.attr().getMode());

    auto inputShape = origOp.input().getType().cast<vpux::NDTypeInterface>().getShape();
    auto outputShape = origOp.output().getType().cast<vpux::NDTypeInterface>().getShape();
    auto scales = SmallVector<double>{
            static_cast<double>(outputShape[Dims4D::Act::N]) / static_cast<double>(inputShape[Dims4D::Act::N]),
            static_cast<double>(outputShape[Dims4D::Act::C]) / static_cast<double>(inputShape[Dims4D::Act::C]),
            static_cast<double>(outputShape[Dims4D::Act::H]) / static_cast<double>(inputShape[Dims4D::Act::H]),
            static_cast<double>(outputShape[Dims4D::Act::W]) / static_cast<double>(inputShape[Dims4D::Act::W])};

    const auto sparseInput = createSparseInput(origOp, rewriter, modeAttr, scales);
    const auto weights = createWeightsConstant(origOp, rewriter, modeAttr, scales);
    const auto weightsShape = weights.getType().cast<vpux::NDTypeInterface>().getShape();
    const auto rawFilterShape = getIntArrayAttr(rewriter, weightsShape);

    const auto ppeTaskAttr = nullptr;

    const auto outputType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto OC = outputType.getShape()[Dims4D::Act::C];
    const auto weightsTableVec = VPU::createWeightsTableData(origOp.input(), origOp.output(), weights, nullptr, OC,
                                                             ppeTaskAttr, _arch, nullptr);
    const auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    auto nceOp = rewriter.create<VPU::NCEInterpolateOp>(origOp->getLoc(), outputType, sparseInput, weights,
                                                        weightsTable, ppeTaskAttr, rawFilterShape,
                                                        /*multi_cluster_strategyAttr=*/nullptr, modeAttr);

    rewriter.replaceOp(origOp, nceOp.output());
    return mlir::success();
}

//
// LowerOpsToSENCEPass
//

class LowerOpsToSENCEPass final : public VPU::LowerOpsToSENCEBase<LowerOpsToSENCEPass> {
public:
    explicit LowerOpsToSENCEPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void LowerOpsToSENCEPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<VPU::InterpolateOp>([&](VPU::InterpolateOp op) {
        return !VPU::NCEInterpolateOp::isSupported(op, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true);
    });
    target.addLegalOp<VPU::NCEInterpolateOp>();
    target.addLegalOp<VPU::StorageElementTableOp>();
    target.addLegalOp<VPU::GroupSparseTensorOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<InterpolateToNCE>(&ctx, arch, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLowerOpsToSENCEPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createLowerOpsToSENCEPass(Logger log) {
    return std::make_unique<LowerOpsToSENCEPass>(log);
}

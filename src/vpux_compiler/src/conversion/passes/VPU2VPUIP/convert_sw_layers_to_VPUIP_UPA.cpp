//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/conversion/rewriters/VPU2VPUIP/sw_rewriter.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// ReverseSequenceRewrite
//

class ReverseSequenceRewrite final : public mlir::OpConversionPattern<VPU::ReverseSequenceOp> {
public:
    ReverseSequenceRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::ReverseSequenceOp>(typeConverter, ctx), _log(log) {
        setDebugName("ReverseSequenceRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ReverseSequenceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReverseSequenceRewrite::matchAndRewrite(VPU::ReverseSequenceOp origOp, OpAdaptor newArgs,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found ReverseSequence Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto origSeqLengthShapeType = origOp.seq_length().getType().cast<mlir::ShapedType>();
    auto newSeqLengthShapeType =
            origSeqLengthShapeType.clone(origSeqLengthShapeType.getShape(), mlir::Float16Type::get(getContext()));
    auto memRefType = typeConverter->convertType(newSeqLengthShapeType);
    auto allocOp = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), memRefType.cast<mlir::MemRefType>());

    auto convertOp = rewriter.create<VPUIP::ConvertUPAOp>(origOp->getLoc(), newArgs.seq_length(), allocOp.memref());

    auto resultBufs = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    rewriter.replaceOpWithNewOp<VPUIP::ReverseSequenceUPAOp>(origOp, newArgs.data(), convertOp.output(), resultBufs[0],
                                                             origOp.seq_axisAttr(), origOp.batch_axisAttr());

    return mlir::success();
}

//
// LSTMCellRewrite
//

class LSTMCellRewrite final : public mlir::OpConversionPattern<VPU::LSTMCellOp> {
public:
    LSTMCellRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::LSTMCellOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::LSTMCellOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMCellRewrite::matchAndRewrite(VPU::LSTMCellOp origOp, OpAdaptor newArgs,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found LSTMCell Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    // Concatenate 'weights' and 'recurrenceWeights' into single buffer

    const auto srcWeights = typeConverter->materializeSourceConversion(rewriter, origOp->getLoc(),
                                                                       origOp.weights().getType(), newArgs.weights());
    const auto srcRecurrenceWeights = typeConverter->materializeSourceConversion(
            rewriter, origOp->getLoc(), origOp.recurrenceWeights().getType(), newArgs.recurrenceWeights());

    auto srcConcatenatedWeights =
            rewriter.create<VPU::ConcatOp>(origOp->getLoc(), mlir::ValueRange{srcWeights, srcRecurrenceWeights}, 1);

    const auto targetConcatenatedWeights = typeConverter->materializeTargetConversion(
            rewriter, origOp->getLoc(), typeConverter->convertType(srcConcatenatedWeights.getType()),
            srcConcatenatedWeights.output());

    auto resultBufs = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    rewriter.replaceOpWithNewOp<VPUIP::LSTMCellUPAOp>(origOp, newArgs.inputData(), newArgs.initialHiddenState(),
                                                      newArgs.initialCellState(), targetConcatenatedWeights,
                                                      newArgs.biases(), resultBufs[0], resultBufs[1]);

    return mlir::success();
}

//
// LSTMSequenceRewrite
//

class LSTMSequenceRewrite final : public mlir::OpConversionPattern<VPU::LSTMSequenceOp> {
public:
    LSTMSequenceRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::LSTMSequenceOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::LSTMSequenceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMSequenceRewrite::matchAndRewrite(VPU::LSTMSequenceOp origOp, OpAdaptor newArgs,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found LSTMSequence Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    // Concatenate 'weights' and 'recurrenceWeights' into single buffer

    const auto srcWeights = typeConverter->materializeSourceConversion(rewriter, origOp->getLoc(),
                                                                       origOp.weights().getType(), newArgs.weights());
    const auto srcRecurrenceWeights = typeConverter->materializeSourceConversion(
            rewriter, origOp->getLoc(), origOp.reccurenceWeights().getType(), newArgs.reccurenceWeights());

    auto srcConcatenatedWeights =
            rewriter.create<VPU::ConcatOp>(origOp->getLoc(), mlir::ValueRange{srcWeights, srcRecurrenceWeights}, 2);

    const auto targetConcatenatedWeights = typeConverter->materializeTargetConversion(
            rewriter, origOp->getLoc(), typeConverter->convertType(srcConcatenatedWeights.getType()),
            srcConcatenatedWeights.output());

    auto resultBufs = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    rewriter.replaceOpWithNewOp<VPUIP::LSTMSequenceUPAOp>(origOp, newArgs.inputData(), newArgs.initialHiddenState(),
                                                          newArgs.initialCellState(), targetConcatenatedWeights,
                                                          newArgs.biases(), resultBufs[0], resultBufs[1], resultBufs[2],
                                                          origOp.sequenceLengthAttr(), origOp.directionAttr());

    return mlir::success();
}

//
// FakeQuantizeRewrite
//

class FakeQuantizeRewrite final : public mlir::OpConversionPattern<VPU::FakeQuantizeOp> {
public:
    FakeQuantizeRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::FakeQuantizeOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::FakeQuantizeOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FakeQuantizeRewrite::matchAndRewrite(VPU::FakeQuantizeOp origOp, OpAdaptor newArgs,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found FakeQuantize Operation '{0}'", origOp->getLoc());

    auto inLowConst = newArgs.input_low().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = newArgs.input_high().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = newArgs.output_low().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = newArgs.output_high().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return matchFailed(rewriter, origOp, "Got non constant parameters");
    }

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, {origOp.output()});

    rewriter.replaceOpWithNewOp<VPUIP::FakeQuantizeUPAOp>(
            origOp, newArgs.input(), outputBuffers[0], origOp.levelsAttr(), inLowConst.getContentAttr(),
            inHighConst.getContentAttr(), outLowConst.getContentAttr(), outHighConst.getContentAttr());

    return mlir::success();
}

//
// FullyConnectedRewrite
//

class FullyConnectedRewrite final : public mlir::OpConversionPattern<VPU::FullyConnectedOp> {
public:
    FullyConnectedRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::FullyConnectedOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::FullyConnectedOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FullyConnectedRewrite::matchAndRewrite(VPU::FullyConnectedOp origOp, OpAdaptor newArgs,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found FullyConnected Operation '{0}'", origOp->getLoc());

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, {origOp.output()});

    if (origOp.bias() == nullptr) {
        rewriter.replaceOpWithNewOp<VPUIP::FullyConnectedUPAOp>(origOp, newArgs.input(), newArgs.weights(), nullptr,
                                                                outputBuffers[0]);
        return mlir::success();
    }

    const auto origBiasType = newArgs.bias().getType().cast<vpux::NDTypeInterface>();

    const auto origBiasShape = origBiasType.getShape().raw();
    VPUX_THROW_UNLESS(origBiasShape[0] == 1, "Biases batch size is not equal 1");

    const std::array<int64_t, 1> newBiasShape = {origBiasShape[1]};
    const auto newBiasType = origBiasType.changeShape(ShapeRef(newBiasShape));

    auto newBias = rewriter.create<VPUIP::GenericReshapeOp>(origOp->getLoc(), newBiasType, newArgs.bias());

    rewriter.replaceOpWithNewOp<VPUIP::FullyConnectedUPAOp>(origOp, newArgs.input(), newArgs.weights(),
                                                            newBias.output(), outputBuffers[0]);

    return mlir::success();
}

//
// RewriteConvolution
//

class RewriteConvolution final : public mlir::OpConversionPattern<VPU::ConvolutionOp> {
public:
    RewriteConvolution(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::ConvolutionOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ConvolutionOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RewriteConvolution::matchAndRewrite(VPU::ConvolutionOp origOp, OpAdaptor newArgs,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Convolution Operation '{0}'", origOp->getLoc());

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, {origOp.output()});

    const int64_t groups = 1;
    if (origOp.bias() == nullptr) {
        rewriter.replaceOpWithNewOp<VPUIP::ConvolutionUPAOp>(origOp, newArgs.input(), newArgs.filter(), nullptr,
                                                             outputBuffers[0], origOp.strides(), origOp.dilations(),
                                                             origOp.pads_begin(), origOp.pads_end(), groups);
        return mlir::success();
    }

    const auto origBiasType = newArgs.bias().getType().cast<vpux::NDTypeInterface>();
    const auto origBiasShape = origBiasType.getShape().raw();

    const std::array<int64_t, 1> newBiasShape = {origBiasShape[1]};
    const auto newBiasType = origBiasType.changeShape(ShapeRef(newBiasShape));
    auto newBias = rewriter.create<VPUIP::GenericReshapeOp>(origOp->getLoc(), newBiasType, newArgs.bias());

    rewriter.replaceOpWithNewOp<VPUIP::ConvolutionUPAOp>(origOp, newArgs.input(), newArgs.filter(), newBias.output(),
                                                         outputBuffers[0], origOp.strides(), origOp.dilations(),
                                                         origOp.pads_begin(), origOp.pads_end(), groups);
    return mlir::success();
}

//
// TopKRewrite
//

class TopKRewrite final : public mlir::OpConversionPattern<VPU::TopKOp> {
public:
    TopKRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::TopKOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::TopKOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TopKRewrite::matchAndRewrite(VPU::TopKOp origOp, OpAdaptor newArgs,
                                                 mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found TopK Operation '{0}'", origOp->getLoc());

    auto outputValuesBuffers =
            allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, {origOp.output_values()});
    auto targetShapesBuffers =
            allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, {origOp.target_shape()});

    rewriter.replaceOpWithNewOp<VPUIP::TopKUPAOp>(origOp, newArgs.input(), newArgs.k(), outputValuesBuffers[0],
                                                  targetShapesBuffers[0], origOp.axis(), origOp.mode(), origOp.sort(),
                                                  origOp.element_type());

    return mlir::success();
}

// RewriteNonMaxSuppression
//

class NonMaxSuppressionRewrite final : public mlir::OpConversionPattern<VPU::NonMaxSuppressionOp> {
public:
    NonMaxSuppressionRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::NonMaxSuppressionOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NonMaxSuppressionOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NonMaxSuppressionRewrite::matchAndRewrite(VPU::NonMaxSuppressionOp origOp, OpAdaptor newArgs,
                                                              mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found NonMaxSuppression Operation '{0}'", origOp->getLoc());

    auto resultBufs = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    rewriter.replaceOpWithNewOp<VPUIP::NonMaxSuppressionUPAOp>(
            origOp, newArgs.in_box_coords(), newArgs.in_box_scores(), resultBufs[0], resultBufs[1], resultBufs[2],
            origOp.box_encodingAttr(), origOp.sort_result_descendingAttr(),
            origOp.max_output_boxes_per_class_valueAttr(), origOp.iou_threshold_valueAttr(),
            origOp.score_threshold_valueAttr(), origOp.soft_nms_sigma_valueAttr());

    _log.trace("Replaced with 'VPUIP.NonMaxSuppressionOp'");

    return mlir::success();
}

//
// ConvertSWLayers2VPUIPUPAPass
//

class ConvertSWLayers2VPUIPUPAPass final : public ConvertSWLayers2VPUIPUPABase<ConvertSWLayers2VPUIPUPAPass> {
public:
    explicit ConvertSWLayers2VPUIPUPAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertSWLayers2VPUIPUPAPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    vpux::BufferizeTypeConverter typeConverter;

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalDialect<Const::ConstDialect>(isLegalOp);
    target.addIllegalDialect<VPU::VPUDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp>();
    target.addLegalOp<mlir::memref::AllocOp>();
    // NCE ops are not handled in this pass
    target.addLegalOp<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp,
                      VPU::NCEEltwiseOp, VPU::NCEPermuteQuantizeOp>();
    target.addLegalOp<VPU::NCEClusterTilingOp, VPU::YieldOp>();
    target.addLegalOp<VPU::DPUWorkloadOp>();
    // ViewLike and other non-SW ops are not handled in this pass
    target.addLegalOp<VPU::CopyOp, VPU::ExpandOp, VPU::StridedSliceOp, VPU::AffineReshapeOp, VPU::ReshapeOp,
                      VPU::SqueezeOp, VPU::UnsqueezeOp, VPU::SliceOp, VPU::SplitOp, VPU::ConcatOp, VPU::PermuteCastOp,
                      VPU::QuantizeCastOp, VPU::DistributedCastOp, VPU::StubOp, VPU::GroupSparseTensorOp,
                      VPU::StorageElementTableOp, VPU::ShapeCastOp, VPU::LayoutCastOp, VPU::WorkloadCastOp>();

    target.addLegalOp<VPUIP::SwKernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });
    vpux::populateBufferizeMaterializationLegality(target);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LayerRewrite>(typeConverter, &ctx, _log);
    patterns.add<ReverseSequenceRewrite>(typeConverter, &ctx, _log);
    patterns.add<LSTMCellRewrite>(typeConverter, &ctx, _log);
    patterns.add<LSTMSequenceRewrite>(typeConverter, &ctx, _log);
    patterns.add<FakeQuantizeRewrite>(typeConverter, &ctx, _log);
    patterns.add<FullyConnectedRewrite>(typeConverter, &ctx, _log);
    patterns.add<RewriteConvolution>(typeConverter, &ctx, _log);
    patterns.add<TopKRewrite>(typeConverter, &ctx, _log);
    patterns.add<NonMaxSuppressionRewrite>(typeConverter, &ctx, _log);

    Const::ConstDialect::populateBufferizePatterns(patterns, typeConverter, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertSWLayers2VPUIPUPAPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertSWLayers2VPUIPUPAPass(Logger log) {
    return std::make_unique<ConvertSWLayers2VPUIPUPAPass>(log);
}

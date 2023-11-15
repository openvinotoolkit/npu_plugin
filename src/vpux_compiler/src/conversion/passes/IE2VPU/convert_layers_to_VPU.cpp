//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// CTCGreedyDecoderSeqLenRewrite
//

class CTCGreedyDecoderSeqLenRewrite final : public mlir::OpRewritePattern<IE::CTCGreedyDecoderSeqLenOp> {
public:
    CTCGreedyDecoderSeqLenRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::CTCGreedyDecoderSeqLenOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::CTCGreedyDecoderSeqLenOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CTCGreedyDecoderSeqLenRewrite::matchAndRewrite(IE::CTCGreedyDecoderSeqLenOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("Found CTCGreedyDecoderSeqLen Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::CTCGreedyDecoderSeqLenOp>(origOp, origOp.input(), origOp.sequenceLength(),
                                                               origOp.blankIndex(), origOp.mergeRepeatedAttr());

    return mlir::success();
}

//
// ProposalRewrite
//

class ProposalRewrite final : public mlir::OpRewritePattern<IE::ProposalOp> {
public:
    ProposalRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ProposalOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ProposalOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ProposalRewrite::matchAndRewrite(IE::ProposalOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Proposal Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::ProposalOp>(origOp, origOp.class_probs(), origOp.bbox_deltas(),
                                                 origOp.image_shape(), origOp.proposal_attrsAttr());
    _log.trace("Replaced with 'VPU.ProposalOp'");

    return mlir::success();
}

//
// SplitRewrite
//

class SplitRewrite final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    SplitRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SplitOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitRewrite::matchAndRewrite(IE::SplitOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Split Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::SplitOp>(origOp, origOp.input(), origOp.axis(), origOp.num_splitsAttr(),
                                              origOp.axis_valueAttr());

    return mlir::success();
}

class StubRewrite final : public mlir::OpRewritePattern<IE::StubOp> {
public:
    StubRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::StubOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::StubOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult StubRewrite::matchAndRewrite(IE::StubOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.debug("Found Stub Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::StubOp>(origOp, origOp.outputs().getTypes(), origOp.inputs());

    return mlir::success();
}

void computeWeightForEmbeddingOp(mlir::MLIRContext* ctx, mlir::RankedTensorType& weightsTensorType,
                                 mlir::DenseElementsAttr& baseAttr, llvm::ArrayRef<int64_t> weightsShape,
                                 vpux::NDTypeInterface inType) {
    // Serialization of optional arguments for sw operators not supported
    // weight tensor is constructed when it is not provided
    const auto iType = inType.getElementType();

    if (iType.isUnsignedInteger(8)) {
        // netPrecision:U8
        weightsTensorType = mlir::RankedTensorType::get(
                weightsShape, mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned));
        baseAttr = mlir::DenseElementsAttr::get(weightsTensorType, (uint8_t)1);

    } else if (iType.isInteger(32)) {
        // netPrecision:int32
        weightsTensorType = mlir::RankedTensorType::get(
                weightsShape, mlir::IntegerType::get(ctx, 32, mlir::IntegerType::SignednessSemantics::Signed));
        baseAttr = mlir::DenseElementsAttr::get(weightsTensorType, 1);

    } else if (iType.isF16()) {
        // netPrecision:float16
        weightsTensorType = mlir::RankedTensorType::get(weightsShape, mlir::Float16Type::get(ctx));
        baseAttr = mlir::DenseElementsAttr::get(weightsTensorType, ngraph::float16(1));
    } else {
        VPUX_THROW("Unsupported element type: {0}", iType);
    }
}

//
// EmbeddingBagOffsetsSumRewriterVPUX30XX
//

class EmbeddingBagOffsetsSumRewriterVPUX30XX final : public mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp> {
public:
    EmbeddingBagOffsetsSumRewriterVPUX30XX(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingBagOffsetsSumOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EmbeddingBagOffsetsSumRewriterVPUX30XX::matchAndRewrite(IE::EmbeddingBagOffsetsSumOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    rewriter.replaceOpWithNewOp<VPU::EmbeddingBagOffsetsSumOp>(
            origOp, origOp.emb_table(), nullptr, nullptr, nullptr, origOp.indices_valueAttr(),
            origOp.offsets_valueAttr(), origOp.default_index_valueAttr(), origOp.per_sample_weights_valueAttr());
    return mlir::success();
}

//
// EmbeddingBagOffsetsSumRewriterVPUX37XX
//

class EmbeddingBagOffsetsSumRewriterVPUX37XX final : public mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp> {
public:
    EmbeddingBagOffsetsSumRewriterVPUX37XX(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingBagOffsetsSumOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EmbeddingBagOffsetsSumRewriterVPUX37XX::matchAndRewrite(IE::EmbeddingBagOffsetsSumOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Found EmbeddingBagOffsetsSumOp Operation '{0}'", origOp->getLoc());

    auto* ctx = origOp->getContext();
    const auto weights = origOp.per_sample_weights();
    if (weights != nullptr) {
        rewriter.replaceOpWithNewOp<VPU::EmbeddingBagOffsetsSumOp>(
                origOp, origOp.emb_table(), origOp.indices(), origOp.offsets(), origOp.per_sample_weights(), nullptr,
                nullptr, origOp.default_index_valueAttr(), nullptr);
        return mlir::success();
    }

    mlir::RankedTensorType weightsTensorType;
    mlir::DenseElementsAttr baseAttr;
    const auto weightsShape = getShape(origOp.indices()).raw();
    const auto inType = origOp.emb_table().getType().cast<NDTypeInterface>();

    computeWeightForEmbeddingOp(ctx, weightsTensorType, baseAttr, weightsShape, inType);

    auto cst = rewriter.create<Const::DeclareOp>(origOp.getLoc(), weightsTensorType, Const::ContentAttr::get(baseAttr));

    rewriter.replaceOpWithNewOp<VPU::EmbeddingBagOffsetsSumOp>(origOp, origOp.emb_table(), origOp.indices(),
                                                               origOp.offsets(), cst.getOutput(), nullptr, nullptr,
                                                               origOp.default_index_valueAttr(), nullptr);

    return mlir::success();
}

//
// RewriteNonMaxSuppression
//

class NonMaxSuppressionRewrite final : public mlir::OpRewritePattern<IE::NonMaxSuppressionOp> {
public:
    NonMaxSuppressionRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::NonMaxSuppressionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::NonMaxSuppressionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NonMaxSuppressionRewrite::matchAndRewrite(IE::NonMaxSuppressionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Found NonMaxSuppression Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::NonMaxSuppressionOp>(
            origOp, origOp.in_box_coords(), origOp.in_box_scores(), origOp.box_encoding(),
            origOp.sort_result_descending(), origOp.max_output_boxes_per_class_valueAttr(),
            origOp.iou_threshold_valueAttr(), origOp.score_threshold_valueAttr(), origOp.soft_nms_sigma_valueAttr());

    _log.trace("Replaced with 'VPU.NonMaxSuppressionOp'");

    return mlir::success();
}

//
// RewriteEmbeddingSegmentsSumVPUX30XX
//

class EmbeddingSegmentsSumRewriterVPUX30XX : public mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp> {
public:
    EmbeddingSegmentsSumRewriterVPUX30XX(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingSegmentsSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EmbeddingSegmentsSumRewriterVPUX30XX::matchAndRewrite(IE::EmbeddingSegmentsSumOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    rewriter.replaceOpWithNewOp<VPU::EmbeddingSegmentsSumOp>(
            origOp, origOp.emb_table(), nullptr, nullptr, nullptr, origOp.indices_valueAttr(),
            origOp.segment_ids_valueAttr(), origOp.num_segments_valueAttr(), origOp.default_index_valueAttr(),
            origOp.per_sample_weights_valueAttr());
    return mlir::success();
}

//
// EmbeddingSegmentsSumRewriterVPUX37XX
//

class EmbeddingSegmentsSumRewriterVPUX37XX : public mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp> {
public:
    EmbeddingSegmentsSumRewriterVPUX37XX(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingSegmentsSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EmbeddingSegmentsSumRewriterVPUX37XX::matchAndRewrite(IE::EmbeddingSegmentsSumOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    auto* ctx = origOp->getContext();
    const auto weights = origOp.per_sample_weights();
    if (weights != nullptr) {
        rewriter.replaceOpWithNewOp<VPU::EmbeddingSegmentsSumOp>(
                origOp, origOp.emb_table(), origOp.indices(), origOp.segment_ids(), origOp.per_sample_weights(),
                nullptr, nullptr, origOp.num_segments_valueAttr(), origOp.default_index_valueAttr(), nullptr);
        return mlir::success();
    }

    mlir::RankedTensorType weightsTensorType;
    mlir::DenseElementsAttr baseAttr;
    const auto weightsShape = getShape(origOp.indices()).raw();
    const auto inType = origOp.emb_table().getType().cast<NDTypeInterface>();

    computeWeightForEmbeddingOp(ctx, weightsTensorType, baseAttr, weightsShape, inType);

    auto cst = rewriter.create<Const::DeclareOp>(origOp.getLoc(), weightsTensorType, Const::ContentAttr::get(baseAttr));

    rewriter.replaceOpWithNewOp<VPU::EmbeddingSegmentsSumOp>(
            origOp, origOp.emb_table(), origOp.indices(), origOp.segment_ids(), cst.getOutput(), nullptr, nullptr,
            origOp.num_segments_valueAttr(), origOp.default_index_valueAttr(), nullptr);
    return mlir::success();
}

//
// RewriteGRUCell
//

class GRUCellRewrite final : public mlir::OpRewritePattern<IE::GRUCellOp> {
public:
    GRUCellRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GRUCellOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GRUCellOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GRUCellRewrite::matchAndRewrite(IE::GRUCellOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found GRUCell Operation '{0}'", origOp->getLoc());

    auto* ctx = origOp->getContext();
    const auto inputShape = getShape(origOp.input_data()).raw();
    const auto batchSize = inputShape[0];
    const auto inputSize = inputShape[1];
    SmallVector<int64_t> newInputShape = {batchSize, 1, inputSize};
    const auto newInputShapeAttr = getIntArrayAttr(ctx, newInputShape);
    auto newInput =
            rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.input_data(), nullptr, false, newInputShapeAttr);

    const auto initialStateShape = getShape(origOp.initial_hidden_state()).raw();
    const auto hiddenSize = initialStateShape[1];
    SmallVector<int64_t> newInitialStateShape = {batchSize, 1, hiddenSize};
    const auto newInitialStateShapeAttr = getIntArrayAttr(ctx, newInitialStateShape);
    auto newInitialState = rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.initial_hidden_state(), nullptr,
                                                           false, newInitialStateShapeAttr);

    SmallVector<int64_t> newWeightsShape = {1, 3 * hiddenSize, inputSize};
    const auto newWeightsShapeAttr = getIntArrayAttr(ctx, newWeightsShape);
    auto newWeights =
            rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.weights(), nullptr, false, newWeightsShapeAttr);

    SmallVector<int64_t> newReWeightsShape = {1, 3 * hiddenSize, hiddenSize};
    const auto newReWeightsShapeAttr = getIntArrayAttr(ctx, newReWeightsShape);
    auto newReWeights = rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.recurrence_weights(), nullptr, false,
                                                        newReWeightsShapeAttr);

    const auto biasesShape = getShape(origOp.biases()).raw();
    SmallVector<int64_t> newBiasesShape = {1, biasesShape[0]};
    const auto newBiasesShapeAttr = getIntArrayAttr(ctx, newBiasesShape);
    auto newBiases =
            rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), origOp.biases(), nullptr, false, newBiasesShapeAttr);

    const auto seqLenAttr = getIntAttr(ctx, 1);
    const auto directionAttr = IE::RNNSequenceDirectionAttr::get(ctx, IE::RNNSequenceDirection::FORWARD);

    auto gruSeq = rewriter.create<VPU::GRUSequenceOp>(
            origOp->getLoc(), newInput, newInitialState, newWeights, newReWeights, newBiases, origOp.hidden_sizeAttr(),
            seqLenAttr, directionAttr, origOp.should_linear_before_resetAttr(), origOp.clipAttr());
    SmallVector<int64_t> newOutputShape = {batchSize, hiddenSize};
    const auto newOutputShapeAttr = getIntArrayAttr(ctx, newOutputShape);
    rewriter.replaceOpWithNewOp<VPU::ReshapeOp>(origOp, gruSeq.output_hidden_state(), nullptr, false,
                                                newOutputShapeAttr);

    return mlir::success();
}

//
// RewriteEmbeddingBagPackedSum
//

class EmbeddingBagPackedSumRewrite final : public mlir::OpRewritePattern<IE::EmbeddingBagPackedSumOp> {
public:
    EmbeddingBagPackedSumRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::EmbeddingBagPackedSumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingBagPackedSumOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EmbeddingBagPackedSumRewrite::matchAndRewrite(IE::EmbeddingBagPackedSumOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Found EmbeddingBagPackedSum Operation '{0}'", origOp->getLoc());

    auto* ctx = origOp->getContext();
    const auto weights = origOp.per_sample_weights();
    if (weights != nullptr) {
        rewriter.replaceOpWithNewOp<VPU::EmbeddingBagPackedSumOp>(origOp, origOp.emb_table(), origOp.indices(),
                                                                  origOp.per_sample_weights());
        return mlir::success();
    }

    mlir::RankedTensorType weightsTensorType;
    mlir::DenseElementsAttr baseAttr;
    const auto weightsShape = getShape(origOp.indices()).raw();
    const auto inType = origOp.emb_table().getType().cast<NDTypeInterface>();

    computeWeightForEmbeddingOp(ctx, weightsTensorType, baseAttr, weightsShape, inType);

    auto cst = rewriter.create<Const::DeclareOp>(origOp.getLoc(), weightsTensorType, Const::ContentAttr::get(baseAttr));
    rewriter.replaceOpWithNewOp<VPU::EmbeddingBagPackedSumOp>(origOp, origOp.emb_table(), origOp.indices(),
                                                              cst.getOutput());
    return mlir::success();
}

//
// InterpolateRewrite
//

// IE.Interpolate -> VPU.Interpolate
class InterpolateRewrite : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    InterpolateRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult InterpolateRewrite::matchAndRewrite(IE::InterpolateOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    rewriter.replaceOpWithNewOp<VPU::InterpolateOp>(
            origOp, origOp.getType(), origOp.input(), origOp.sizes(), origOp.scales(), origOp.axes(),
            origOp.sizes_attrAttr(), origOp.scales_attrAttr(), origOp.axes_attrAttr(), origOp.tile_offset_attrAttr(),
            origOp.initial_input_dims_attrAttr(), origOp.initial_output_dims_attrAttr(),
            /*initial_input_offset_attr=*/nullptr, /*initial_output_offset_attr=*/nullptr,
            /*multiClusterStrategy=*/nullptr, origOp.attrAttr());
    return mlir::success();
}

//
// TopKRewrite
//

class TopKRewrite final : public mlir::OpRewritePattern<IE::TopKOp> {
public:
    TopKRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TopKOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TopKOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TopKRewrite::matchAndRewrite(IE::TopKOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found TopK Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPU::TopKOp>(origOp, origOp.input(), origOp.k(), origOp.k_valueAttr(), origOp.axis(),
                                             origOp.mode(), origOp.sort(), origOp.element_type(), nullptr);

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/conversion/convert_layers_to_VPU.hpp.inc>

//
// ConvertLayers2VPUPass
//

class ConvertLayers2VPUPass final : public ConvertLayers2VPUBase<ConvertLayers2VPUPass> {
public:
    explicit ConvertLayers2VPUPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertLayers2VPUPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto arch = VPU::getArch(func);

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<IE::IEDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<CTCGreedyDecoderSeqLenRewrite>(&ctx, _log);
    patterns.add<ProposalRewrite>(&ctx, _log);
    patterns.add<SplitRewrite>(&ctx, _log);
    patterns.add<StubRewrite>(&ctx, _log);
    patterns.add<NonMaxSuppressionRewrite>(&ctx, _log);
    patterns.add<InterpolateRewrite>(&ctx, _log);

    if (arch == VPU::ArchKind::VPUX37XX) {
        patterns.add<EmbeddingSegmentsSumRewriterVPUX37XX>(&ctx, _log);
        patterns.add<EmbeddingBagOffsetsSumRewriterVPUX37XX>(&ctx, _log);
    } else {
        patterns.add<EmbeddingSegmentsSumRewriterVPUX30XX>(&ctx, _log);
        patterns.add<EmbeddingBagOffsetsSumRewriterVPUX30XX>(&ctx, _log);
    }

    patterns.add<GRUCellRewrite>(&ctx, _log);
    patterns.add<EmbeddingBagPackedSumRewrite>(&ctx, _log);
    patterns.add<TopKRewrite>(&ctx, _log);
    populateWithGenerated(patterns);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2VPUPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertLayers2VPUPass(Logger log) {
    return std::make_unique<ConvertLayers2VPUPass>(log);
}

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <numeric>
#include <vpux/compiler/conversion.hpp>
#include <vpux/utils/core/error.hpp>

using namespace vpux;

namespace {

//
// Helper functions
//

auto getNumPriors(VPU::DetectionOutputOp origOp) {
    const auto priorBoxSize = origOp.attr().getNormalized().getValue() ? 4 : 5;
    const auto numPriors = getShape(origOp.in_proposals())[Dim(2)] / priorBoxSize;
    return numPriors;
}

mlir::Value createReshape(mlir::PatternRewriter& rewriter, mlir::Value tensor, ArrayRef<int64_t> newShape) {
    const auto newShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
    auto reshape = rewriter.create<VPU::ReshapeOp>(tensor.getLoc(), tensor, nullptr, false, newShapeAttr);
    return reshape.output();
}

mlir::Value reshapeToSeparateClassAndPriors(mlir::PatternRewriter& rewriter, mlir::Value input, int numPriors) {
    const auto inputShape = getShape(input);
    const auto batch = inputShape.front();
    const auto width = inputShape.back();

    VPUX_THROW_WHEN(
            (width % numPriors) != 0,
            "DetectionOutput input tensor with shape {0} has width={1}, that must be divisible by numPriors={2}",
            inputShape, width, numPriors);

    const auto newShape = SmallVector<int64_t>{batch, numPriors, width / numPriors};
    return createReshape(rewriter, input, newShape);
}

mlir::Value reshapeLogitsTo4D(mlir::PatternRewriter& rewriter, mlir::Value input, int numPriors) {
    const auto inputShape = getShape(input);
    VPUX_THROW_UNLESS(inputShape.size() == 2, "boxLogits shape must be 2D");

    const auto batch = inputShape[Dim(0)];
    const auto width = inputShape[Dim(1)];
    const auto boxSize = 4;

    VPUX_THROW_UNLESS((width % (boxSize * numPriors)) == 0,
                      "DetectionOutput boxLogits tensor with shape {0} has width={1}, that must be divisible by 4 * "
                      "numPriors={2}",
                      inputShape, width, boxSize * numPriors);

    const auto numLocClasses = width / (boxSize * numPriors);
    const auto newShape = SmallVector<int64_t>{batch, numPriors, numLocClasses, boxSize};
    return createReshape(rewriter, input, newShape);
}

mlir::Value transposeHW(mlir::PatternRewriter& rewriter, mlir::Value tensor) {
    const auto ctx = rewriter.getContext();

    const auto shape = getShape(tensor);
    VPUX_THROW_UNLESS(shape.size() >= 2, "Can't transpose 1D tensor");

    const auto dimsOrder = DimsOrder::fromValue(tensor);
    const auto orderMap = dimsOrder.toAffineMap(ctx);
    VPUX_THROW_UNLESS(orderMap.isIdentity(), "Expected identity ordered tensor to be transposed");

    auto memPermutation = dimsOrder.toPermutation();
    std::swap(*memPermutation.rbegin(), *std::next(memPermutation.rbegin()));
    const auto permutationMap = DimsOrder::fromPermutation(memPermutation).toAffineMap(ctx);

    return rewriter.create<VPU::MemPermuteOp>(tensor.getLoc(), tensor, orderMap, permutationMap);
}

mlir::LogicalResult checkSupportedArguments(VPU::DetectionOutputOp origOp) {
    const auto attrs = origOp.attr();

    const auto normalized = origOp.attr().getNormalized().getValue();
    const auto varianceEncodedItTarget = origOp.attr().getVarianceEncodedInTarget().getValue();
    if (!normalized && !varianceEncodedItTarget) {
        return errorAt(origOp,
                       "DetectionOutput: undefined case - normalized == false && varianceEncodedItTarget == false");
    }

    const auto keepTopKArray = origOp.attr().getKeepTopK().getValue();
    if (keepTopKArray.size() != 1) {
        return errorAt(origOp, "DetectionOutput: keepTopK array size={0} is not supported", keepTopKArray.size());
    }

    const auto hasAdditionalInputs = (origOp->getNumOperands() > 3);
    if (hasAdditionalInputs) {
        return errorAt(origOp, "DetectionOutput: only 3 inputs is supported");
    }

    const auto codeType = attrs.getCodeType().getValue();
    if (codeType != IE::DetectionOutputCodeType::CENTER_SIZE) {
        return errorAt(origOp, "DetectionOutput: only CodeType::CENTER_SIZE is supported");
    }

    const auto mxNetNms = (attrs.getDecreaseLabelId().getValue() == true);
    if (mxNetNms) {
        return errorAt(origOp, "DetectionOutput: decreaseLabelId == true is not supported");
    }

    return mlir::success();
}

mlir::Value transposeBoxes(mlir::PatternRewriter& rewriter, mlir::Value boxes4D) {
    const auto ctx = rewriter.getContext();

    // keep default 4D order because this is the order that will be used throughout
    // pretend that the data was never reordered in the first place
    const auto defaultOrder = DimsOrder::NCHW.toAffineMap(ctx);
    const auto permutationMap = DimsOrder::NHCW.toAffineMap(ctx);  // swap Priors and Classes

    return rewriter.create<VPU::MemPermuteOp>(boxes4D.getLoc(), boxes4D, defaultOrder, permutationMap);
}

//
// DetectionOutputDecompositionPass
//

class DetectionOutputDecomposition final : public mlir::OpRewritePattern<VPU::DetectionOutputOp> {
public:
    DetectionOutputDecomposition(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::DetectionOutputOp>(ctx), _log(log) {
        setDebugName("DetectionOutputDecomposition");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::DetectionOutputOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DetectionOutputDecomposition::matchAndRewrite(VPU::DetectionOutputOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto loc = origOp->getLoc();
    const auto supported = checkSupportedArguments(origOp);
    if (!mlir::succeeded(supported)) {
        return supported;
    }

    auto priors = origOp.in_proposals();
    const auto normalized = origOp.attr().getNormalized().getValue();
    if (!normalized) {
        const auto inputWidth = origOp.attr().getInputWidth().getInt();
        const auto inputHeight = origOp.attr().getInputHeight().getInt();
        priors = rewriter.create<VPU::DetectionOutputNormalizeOp>(origOp->getLoc(), priors, inputWidth, inputHeight);
    }

    const auto numPriors = getNumPriors(origOp);
    const auto boxLogits = reshapeLogitsTo4D(rewriter, origOp.in_box_logits(), numPriors);
    const auto boxLogitsTransposed = transposeBoxes(rewriter, boxLogits);

    const auto classPredictions = reshapeToSeparateClassAndPriors(rewriter, origOp.in_class_preds(), numPriors);

    const auto codeType = origOp.attr().getCodeType().getValue();
    const auto clipBeforeNms = origOp.attr().getClipBeforeNms().getValue();
    auto decodeBoxes = rewriter.create<VPU::DetectionOutputDecodeBoxesOp>(
            appendLoc(loc, "decodeBoxes"), boxLogitsTransposed, priors, codeType, clipBeforeNms);

    const auto decodeBoxesShape = getShape(decodeBoxes).toValues();
    const auto newShape = SmallVector<int64_t>{decodeBoxesShape[Dim(0)], decodeBoxesShape[Dim(1)],
                                               decodeBoxesShape[Dim(2)] * decodeBoxesShape[Dim(3)]};
    auto decodeBoxes3D = createReshape(rewriter, decodeBoxes, newShape);

    const auto transposedClassPredictions = transposeHW(rewriter, classPredictions);
    const auto confidenceThreshold = origOp.attr().getConfidenceThreshold();
    const auto topK = origOp.attr().getTopK();
    const auto backgroundId = origOp.attr().getBackgroundLabelId();
    auto sortTopK = rewriter.create<VPU::DetectionOutputSortTopKOp>(
            appendLoc(loc, "sortTopK"), transposedClassPredictions, confidenceThreshold, topK, backgroundId);

    auto selectBoxes = rewriter.create<VPU::DetectionOutputSelectBoxesOp>(
            appendLoc(loc, "selectBoxes"), decodeBoxes3D, sortTopK.out_indices(),  // decodeBoxes.out_decoded_boxes()
            sortTopK.out_sizes(), topK);

    const auto nmsThreshold = origOp.attr().getNmsThreshold();
    auto nmsCaffe = rewriter.create<VPU::DetectionOutputNmsCaffeOp>(
            appendLoc(loc, "nmsCaffe"), sortTopK.out_top_k_confidence(), selectBoxes.out_boxes(), sortTopK.out_sizes(),
            nmsThreshold);

    const auto keepTopKValue = origOp.attr().getKeepTopK()[0].cast<mlir::IntegerAttr>().getInt();
    const auto keepTopK = getIntAttr(rewriter.getContext(), keepTopKValue);
    const auto clipAfterNms = origOp.attr().getClipAfterNms();
    rewriter.replaceOpWithNewOp<VPU::DetectionOutputCollectResultsOp>(
            origOp, nmsCaffe.out_confidence(), nmsCaffe.out_boxes(), nmsCaffe.out_sizes(), keepTopK, clipAfterNms);

    return mlir::success();
}

class DetectionOutputDecompositionPass final :
        public VPU::DetectionOutputDecompositionBase<DetectionOutputDecompositionPass> {
public:
    explicit DetectionOutputDecompositionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DetectionOutputDecompositionPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPU::DetectionOutputOp>();
    target.addLegalOp<VPU::DetectionOutputNormalizeOp>();
    target.addLegalOp<VPU::DetectionOutputDecodeBoxesOp>();
    target.addLegalOp<VPU::DetectionOutputSortTopKOp>();
    target.addLegalOp<VPU::DetectionOutputSelectBoxesOp>();
    target.addLegalOp<VPU::DetectionOutputNmsCaffeOp>();
    target.addLegalOp<VPU::DetectionOutputCollectResultsOp>();

    target.addLegalOp<VPU::ReshapeOp>();
    target.addLegalOp<VPU::MemPermuteOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DetectionOutputDecomposition>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDetectionOutputDecompositionPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createDetectionOutputDecompositionPass(Logger log) {
    return std::make_unique<DetectionOutputDecompositionPass>(log);
}

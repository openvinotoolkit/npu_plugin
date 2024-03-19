//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <functional>
#include <vector>

using namespace vpux;

namespace {
class WeightsDequantizeToFakeQuantizeRewriter final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    WeightsDequantizeToFakeQuantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        setDebugName("WeightsDequantizeToFakeQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter&) const final;

private:
    mlir::FailureOr<std::tuple<Const::ContentAttr, Const::ContentAttr, mlir::RankedTensorType>> applyTransformation(
            Const::ContentAttr inLowContentAttr, Const::ContentAttr inHighContentAttr,
            Const::ContentAttr transformContentAttr, const std::function<float(float, float)>& bFunc) const;
    Logger _log;
};

mlir::FailureOr<std::tuple<Const::ContentAttr, Const::ContentAttr, mlir::RankedTensorType>>
WeightsDequantizeToFakeQuantizeRewriter::applyTransformation(Const::ContentAttr inLowContentAttr,
                                                             Const::ContentAttr inHighContentAttr,
                                                             Const::ContentAttr transformContentAttr,
                                                             const std::function<float(float, float)>& bFunc) const {
    auto transformShape = transformContentAttr.getType().getShape();
    auto inLowShape = inLowContentAttr.getType().getShape();

    // If inLowCst and inHighCst have different rank then the transformCst and the rank of one of the input const or
    // transform const is 1 and the dimension 0 is one align the rank by inserting 1 values to the constant shape else
    // return failure
    if (transformShape.size() != inLowShape.size()) {
        if (transformShape.size() == 1 && transformShape[Dim(0)] == 1) {
            SmallVector<int64_t> newTransformShape(inLowShape.size(), 1);
            transformShape = ShapeRef(newTransformShape);
        } else if (inLowShape.size() == 1 && inLowShape[Dim(0)] == 1) {
            SmallVector<int64_t> newInLowShape(transformShape.size(), 1);
            inLowShape = ShapeRef(newInLowShape);
        } else {
            _log.nest().trace("The transform rank {0} is not equal with inLow rank {1}", transformShape.size(),
                              inLowShape.size());
            return mlir::failure();
        }
    }

    // Align inLowCst/inHighCst and tranformCst shapes
    for (size_t i = 0; i < transformShape.size(); i++) {
        if (transformShape[Dim(i)] == 1 && inLowShape[Dim(i)] > 1) {
            transformContentAttr = transformContentAttr.broadcast(Dim(i), inLowShape[Dim(i)]);
        } else if (inLowShape[Dim(i)] == 1 && transformShape[Dim(i)] > 1) {
            inLowContentAttr = inLowContentAttr.broadcast(Dim(i), transformShape[Dim(i)]);
            inHighContentAttr = inHighContentAttr.broadcast(Dim(i), transformShape[Dim(i)]);
        } else if (transformShape[Dim(i)] > 1 && inLowShape[Dim(i)] > 1 &&
                   transformShape[Dim(i)] != inLowShape[Dim(i)]) {
            _log.nest().nest().trace("Cannot broadcast sizes inLow/inHigh dim = {0} and transform dim = {1}",
                                     inLowShape[Dim(i)], transformShape[Dim(i)]);
            return mlir::failure();
        }
    }

    // Get content of broadcasted input low, input high and zero points constants
    const auto broadcastedTransformContent = transformContentAttr.fold();
    const auto broadcastedInLowContent = inLowContentAttr.fold();
    const auto broadcastedInHighContent = inHighContentAttr.fold();
    const auto broadcastedTransformBaseElemType = broadcastedTransformContent.getType().getElementType();
    const auto broadcastedInLowValues = to_small_vector(broadcastedInLowContent.getValues<float>());
    const auto broadcastedInHighValues = to_small_vector(broadcastedInHighContent.getValues<float>());
    const auto broadcastedTransformValues = to_small_vector(broadcastedTransformContent.getValues<float>());

    auto outLowValues = SmallVector<float>(broadcastedTransformValues.size(), 0);
    auto outHighValues = SmallVector<float>(broadcastedTransformValues.size(), 0);
    loop_1d(LoopExecPolicy::Parallel, broadcastedTransformValues.size(), [&](size_t i) {
        outLowValues[i] = bFunc(broadcastedInLowValues[i], broadcastedTransformValues[i]);
        outHighValues[i] = bFunc(broadcastedInHighValues[i], broadcastedTransformValues[i]);
    });

    auto outConstShape = inLowContentAttr.getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    auto outStorageType = mlir::RankedTensorType::get(outConstShape.raw(), broadcastedTransformBaseElemType);
    const auto outLowDenseElementVal = wrapArrayRef(outStorageType, outLowValues);
    const auto outHighDenseElementVal = wrapArrayRef(outStorageType, outHighValues);
    auto outInLowContentAttr = Const::ContentAttr::get(outLowDenseElementVal);
    auto outInHighContentAttr = Const::ContentAttr::get(outHighDenseElementVal);
    return std::tuple<Const::ContentAttr, Const::ContentAttr, mlir::RankedTensorType>(
            outInLowContentAttr, outInHighContentAttr, outStorageType);
}

mlir::LogicalResult WeightsDequantizeToFakeQuantizeRewriter::matchAndRewrite(IE::MultiplyOp origOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Multiply constant {0} at `{1}`.", origOp->getName(), origOp->getLoc());

    //
    // Pattern matching conditions
    //

    // +----------------------------------------------------------------+
    // | Weights Const - i8 with transformations                        |
    // | [#const.ConvertElemType<f16>] || [#const.ConvertElemType<f32>] |
    // +----------------------------------------------------------------+
    //           |
    //           |      +-----------------+
    //           |      | ZeroPoint Const |
    //           |      +-----------------+
    //           |           |
    //        +-------------------+
    //        | Optional Subtract |
    //        +-------------------+
    //                  |
    //                  |   +-------------+
    //                  |   | Scale Const |
    //                  |   +-------------+
    //                  |         |
    //                 +-----------+
    //                 |  Multiply |
    //                 +-----------+
    //
    // Subtract operation is optional in the dequantization pattern
    mlir::Operation* lastOp = origOp.getOperation();
    auto maybeSubtract = origOp.getInput1().getDefiningOp<IE::SubtractOp>();
    if (maybeSubtract != nullptr) {
        lastOp = maybeSubtract.getOperation();
    }

    auto maybeWeightsCst = lastOp->getOperand(0).getDefiningOp<Const::DeclareOp>();
    if (maybeWeightsCst == nullptr) {
        _log.nest().trace("Weights constant not found");
        return mlir::failure();
    }

    const auto weightsContentAttr = maybeWeightsCst.getContentAttr();
    const auto weightsBaseVals = weightsContentAttr.getBaseContent();
    const auto weightsBaseElemType = weightsBaseVals.getShapedType().getElementType();
    // The only supported weights data type is I8
    if (!weightsBaseElemType.isSignedInteger(8)) {
        _log.nest().trace("Weights data type {0} is not supported.", weightsBaseElemType);
        return mlir::failure();
    }

    // Pattern matched successfully

    //
    // Compute input low, input high constants of FakeQuantize
    //
    const auto weightsContent = maybeWeightsCst.getContent();
    // Convert weights values from int8 to float
    // This should ensure good precision for possible further operation done on weights along the compilation flow
    const auto weightsContentType = weightsContent.getType();
    const auto weightsBufferSize = checked_cast<size_t>(weightsContentType.getNumElements());
    if (weightsBufferSize == 0) {
        _log.nest().trace("Weights constant is empty");
        return mlir::failure();
    }
    const auto weightsElementType = weightsContentType.getElementType();
    const auto weightsBufferByteSize = checked_cast<size_t>(weightsContentType.getTotalAllocSize().count());
    std::vector<float> fp32TempWeightsBuffer;
    std::vector<float16> fp16TempWeightsBuffer;
    float weightsMinimum = 0.f;
    if (weightsElementType.isF16()) {
        fp16TempWeightsBuffer.resize(weightsBufferSize);
        weightsContent.copyTo(
                MutableArrayRef(reinterpret_cast<char*>(fp16TempWeightsBuffer.data()), weightsBufferByteSize));
        weightsMinimum =
                static_cast<float>(*std::min_element(fp16TempWeightsBuffer.begin(), fp16TempWeightsBuffer.end()));
    } else if (weightsElementType.isF32()) {
        fp32TempWeightsBuffer.resize(weightsBufferSize);
        weightsContent.copyTo(
                MutableArrayRef(reinterpret_cast<char*>(fp32TempWeightsBuffer.data()), weightsBufferByteSize));
        weightsMinimum = *std::min_element(fp32TempWeightsBuffer.begin(), fp32TempWeightsBuffer.end());
    } else {
        _log.nest().trace("Weights element type must be FP16 or FP32 but got {0}", weightsElementType);
        return mlir::failure();
    }

    int64_t levels = (isFloatEqual(weightsMinimum, static_cast<float>(-128))) ? 256 : 255;
    int64_t inLow = -(levels / 2);
    int64_t inHigh = levels + inLow - 1;

    const auto weightsConstantRank = getShape(maybeWeightsCst).size();
    SmallVector<int64_t> inCstShape = SmallVector<int64_t>(weightsConstantRank, 1);
    const auto weightsElemType = weightsContentAttr.getType().dyn_cast<vpux::NDTypeInterface>().getElementType();
    const auto inStorageType = mlir::RankedTensorType::get(inCstShape, weightsElemType);
    const auto inLowDenseElementVal = wrapData(inStorageType, float(inLow));
    const auto inHighDenseElementVal = wrapData(inStorageType, float(inHigh));
    auto inLowContentAttr = Const::ContentAttr::get(inLowDenseElementVal);
    auto inHighContentAttr = Const::ContentAttr::get(inHighDenseElementVal);
    auto inLowConst = rewriter.create<Const::DeclareOp>(origOp->getLoc(), inStorageType, inLowContentAttr);
    auto inHighConst = rewriter.create<Const::DeclareOp>(origOp->getLoc(), inStorageType, inHighContentAttr);

    //
    // Compute output low and output high constants of FakeQuantize
    //

    // Apply zero point
    auto outLowContentAttr = inLowContentAttr;
    auto outHighContentAttr = inHighContentAttr;
    auto outStorageType = inStorageType;
    if (maybeSubtract != nullptr) {
        auto zeroPointCst = maybeSubtract.getInput2().getDefiningOp<Const::DeclareOp>();
        if (zeroPointCst == nullptr) {
            return mlir::failure();
        }

        auto zeroPointContentAttr = zeroPointCst.getContentAttr();
        auto applySubtract =
                applyTransformation(outLowContentAttr, outHighContentAttr, zeroPointContentAttr, std::minus<float>());
        if (mlir::failed(applySubtract)) {
            return mlir::failure();
        }

        std::tie(outLowContentAttr, outHighContentAttr, outStorageType) = applySubtract.value();
    }

    // Apply scale
    auto scaleCst = origOp.getInput2().getDefiningOp<Const::DeclareOp>();
    if (scaleCst == nullptr) {
        return mlir::failure();
    }

    auto scaleContentAttr = scaleCst.getContentAttr();
    auto applyMultiply =
            applyTransformation(outLowContentAttr, outHighContentAttr, scaleContentAttr, std::multiplies<float>());
    if (mlir::failed(applyMultiply)) {
        return mlir::failure();
    }

    std::tie(outLowContentAttr, outHighContentAttr, outStorageType) = applyMultiply.value();

    auto outLowConst = rewriter.create<Const::DeclareOp>(origOp->getLoc(), outStorageType, outLowContentAttr);
    auto outHighConst = rewriter.create<Const::DeclareOp>(origOp->getLoc(), outStorageType, outHighContentAttr);

    // Create the FakeQuantize to replace the weights dequantize pattern
    const auto broadCastAttr = IE::AutoBroadcastTypeAttr::get(origOp.getContext(), IE::AutoBroadcastType::NUMPY);
    const auto levelsAttr = getIntAttr(origOp.getContext(), levels);
    mlir::DenseElementsAttr weightsDenseAttr;
    auto weightsRankedTensorType = weightsContentType.cast<mlir::RankedTensorType>();
    if (weightsElementType.isF16()) {
        weightsDenseAttr = mlir::DenseElementsAttr::getFromRawBuffer(
                weightsRankedTensorType,
                ArrayRef(reinterpret_cast<char*>(fp16TempWeightsBuffer.data()), weightsBufferByteSize));
    } else if (weightsElementType.isF32()) {
        weightsDenseAttr = mlir::DenseElementsAttr::getFromRawBuffer(
                weightsRankedTensorType,
                ArrayRef(reinterpret_cast<char*>(fp32TempWeightsBuffer.data()), weightsBufferByteSize));
    } else {
        VPUX_THROW("Unsupported weights element type '{0}'", weightsElementType);
    }

    auto f32WeightsConst = rewriter.create<Const::DeclareOp>(origOp.getLoc(), weightsContentAttr.getType(),
                                                             Const::ContentAttr::get(weightsDenseAttr));
    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(origOp, f32WeightsConst, inLowConst, inHighConst, outLowConst,
                                                    outHighConst, levelsAttr, broadCastAttr);
    return mlir::success();
}

class WeightsDequantizeToFakeQuantizePass final :
        public IE::WeightsDequantizeToFakeQuantizeBase<WeightsDequantizeToFakeQuantizePass> {
public:
    explicit WeightsDequantizeToFakeQuantizePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void WeightsDequantizeToFakeQuantizePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<WeightsDequantizeToFakeQuantizeRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createWeightsDequantizeToFakeQuantizePass
//
std::unique_ptr<mlir::Pass> vpux::IE::createWeightsDequantizeToFakeQuantizePass(Logger log) {
    return std::make_unique<WeightsDequantizeToFakeQuantizePass>(log);
}

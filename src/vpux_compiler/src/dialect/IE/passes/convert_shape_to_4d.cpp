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

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

namespace {

constexpr size_t TARGET_TENSOR_DIM = 4;

//
// ConvertShapeTo4DPass
//

class ConvertShapeTo4DPass final : public IE::ConvertShapeTo4DBase<ConvertShapeTo4DPass> {
public:
    explicit ConvertShapeTo4DPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    static SmallVector<int64_t> getNewShape(mlir::ShapedType tensor);
    static IE::ReshapeOp addReshapeOperation(mlir::Operation* origOp, mlir::Value input, mlir::ArrayRef<int64_t> shape,
                                             mlir::PatternRewriter& rewriter);

public:
    class ClampOpConverter;
    class EluOpConverter;
    class ReluOpConverter;
    class SigmoidOpConverter;
    class HSwishOpConverter;
    class TanhOpConverter;
    class FakeQuantizeOpConverter;
    class ScaleShiftOpConverter;
    class MultiplyOpConverter;
    class AddOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ClampOpConverter
//

class ConvertShapeTo4DPass::ClampOpConverter final : public mlir::OpRewritePattern<IE::ClampOp> {
public:
    ClampOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ClampOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertShapeTo4DPass::ClampOpConverter::matchAndRewrite(IE::ClampOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp.getLoc());

    auto opType = origOp.getType();
    auto newShape = getNewShape(opType);
    if (newShape.empty()) {
        return mlir::failure();
    }

    auto reshapeBefore = addReshapeOperation(origOp, origOp.input(), makeArrayRef(newShape), rewriter);

    auto newClampOp = rewriter.create<IE::ClampOp>(origOp->getLoc(), reshapeBefore, origOp.minAttr(), origOp.maxAttr());

    auto reshapeAfter = addReshapeOperation(origOp, newClampOp.output(), opType.getShape(), rewriter);
    rewriter.replaceOp(origOp, reshapeAfter.output());

    return mlir::success();
}

//
// EluOpConverter
//

class ConvertShapeTo4DPass::EluOpConverter final : public mlir::OpRewritePattern<IE::EluOp> {
public:
    EluOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::EluOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EluOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertShapeTo4DPass::EluOpConverter::matchAndRewrite(IE::EluOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp.getLoc());

    auto opType = origOp.getType();
    auto newShape = getNewShape(opType);
    if (newShape.empty()) {
        return mlir::failure();
    }

    auto reshapeBefore = addReshapeOperation(origOp, origOp.input(), makeArrayRef(newShape), rewriter);

    auto newEluOp = rewriter.create<IE::EluOp>(origOp->getLoc(), reshapeBefore, origOp.xAttr());

    auto reshapeAfter = addReshapeOperation(origOp, newEluOp.output(), opType.getShape(), rewriter);
    rewriter.replaceOp(origOp, reshapeAfter.output());

    return mlir::success();
}

//
// ReluOpConverter
//

class ConvertShapeTo4DPass::ReluOpConverter final : public mlir::OpRewritePattern<IE::ReLUOp> {
public:
    ReluOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReLUOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ReLUOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertShapeTo4DPass::ReluOpConverter::matchAndRewrite(IE::ReLUOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp.getLoc());

    auto opType = origOp.getType();
    auto newShape = getNewShape(opType);
    if (newShape.empty()) {
        return mlir::failure();
    }

    auto reshapeBefore = addReshapeOperation(origOp, origOp.input(), makeArrayRef(newShape), rewriter);

    auto newReluOp = rewriter.create<IE::ReLUOp>(origOp->getLoc(), reshapeBefore);

    auto reshapeAfter = addReshapeOperation(origOp, newReluOp.output(), opType.getShape(), rewriter);
    rewriter.replaceOp(origOp, reshapeAfter.output());

    return mlir::success();
}

//
// SigmoidOpConverter
//

class ConvertShapeTo4DPass::SigmoidOpConverter final : public mlir::OpRewritePattern<IE::SigmoidOp> {
public:
    SigmoidOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SigmoidOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SigmoidOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertShapeTo4DPass::SigmoidOpConverter::matchAndRewrite(IE::SigmoidOp origOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp.getLoc());

    auto opType = origOp.getType();
    auto newShape = getNewShape(opType);
    if (newShape.empty()) {
        return mlir::failure();
    }

    auto reshapeBefore = addReshapeOperation(origOp, origOp.input(), makeArrayRef(newShape), rewriter);

    auto newSigmoidOp = rewriter.create<IE::SigmoidOp>(origOp->getLoc(), reshapeBefore);

    auto reshapeAfter = addReshapeOperation(origOp, newSigmoidOp.output(), opType.getShape(), rewriter);
    rewriter.replaceOp(origOp, reshapeAfter.output());

    return mlir::success();
}

//
// HSwishOpConverter
//

class ConvertShapeTo4DPass::HSwishOpConverter final : public mlir::OpRewritePattern<IE::HSwishOp> {
public:
    HSwishOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::HSwishOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::HSwishOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertShapeTo4DPass::HSwishOpConverter::matchAndRewrite(IE::HSwishOp origOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp.getLoc());

    auto opType = origOp.getType();
    auto newShape = getNewShape(opType);
    if (newShape.empty()) {
        return mlir::failure();
    }

    auto reshapeBefore = addReshapeOperation(origOp, origOp.input(), makeArrayRef(newShape), rewriter);

    auto newHSwishOp = rewriter.create<IE::HSwishOp>(origOp->getLoc(), reshapeBefore);

    auto reshapeAfter = addReshapeOperation(origOp, newHSwishOp.output(), opType.getShape(), rewriter);
    rewriter.replaceOp(origOp, reshapeAfter.output());

    return mlir::success();
}

//
// TanhOpConverter
//

class ConvertShapeTo4DPass::TanhOpConverter final : public mlir::OpRewritePattern<IE::TanhOp> {
public:
    TanhOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TanhOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TanhOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertShapeTo4DPass::TanhOpConverter::matchAndRewrite(IE::TanhOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp.getLoc());

    auto opType = origOp.getType();
    auto newShape = getNewShape(opType);
    if (newShape.empty()) {
        return mlir::failure();
    }

    auto reshapeBefore = addReshapeOperation(origOp, origOp.input(), makeArrayRef(newShape), rewriter);

    auto newTanhOp = rewriter.create<IE::TanhOp>(origOp->getLoc(), reshapeBefore);

    auto reshapeAfter = addReshapeOperation(origOp, newTanhOp.output(), opType.getShape(), rewriter);
    rewriter.replaceOp(origOp, reshapeAfter.output());

    return mlir::success();
}

//
// FakeQuantizeOpConverter
//

class ConvertShapeTo4DPass::FakeQuantizeOpConverter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    FakeQuantizeOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertShapeTo4DPass::FakeQuantizeOpConverter::matchAndRewrite(
        IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp.getLoc());

    auto opType = origOp.getType();
    auto newShape = getNewShape(opType);
    auto newInputLowShape = getNewShape(origOp.input_low().getType().dyn_cast<mlir::ShapedType>());
    auto newInputHighShape = getNewShape(origOp.input_high().getType().dyn_cast<mlir::ShapedType>());
    auto newOutputLowShape = getNewShape(origOp.output_low().getType().dyn_cast<mlir::ShapedType>());
    auto newOutputHighShape = getNewShape(origOp.output_high().getType().dyn_cast<mlir::ShapedType>());
    if (newShape.empty() || newInputLowShape.empty() || newInputHighShape.empty() || newOutputLowShape.empty() ||
        newOutputHighShape.empty()) {
        return mlir::failure();
    }

    auto reshapeBefore = addReshapeOperation(origOp, origOp.input(), makeArrayRef(newShape), rewriter);
    auto reshapedInputLow = addReshapeOperation(origOp, origOp.input_low(), makeArrayRef(newInputLowShape), rewriter);
    auto reshapedInputHigh =
            addReshapeOperation(origOp, origOp.input_high(), makeArrayRef(newInputHighShape), rewriter);
    auto reshapedOutputLow =
            addReshapeOperation(origOp, origOp.output_low(), makeArrayRef(newOutputLowShape), rewriter);
    auto reshapedOutputHigh =
            addReshapeOperation(origOp, origOp.output_high(), makeArrayRef(newOutputHighShape), rewriter);

    auto newFakeQuantizeOp = rewriter.create<IE::FakeQuantizeOp>(
            origOp->getLoc(), reshapeBefore, reshapedInputLow, reshapedInputHigh, reshapedOutputLow, reshapedOutputHigh,
            origOp.levelsAttr(), origOp.auto_broadcastAttr());

    auto reshapeAfter = addReshapeOperation(origOp, newFakeQuantizeOp.output(), opType.getShape(), rewriter);
    rewriter.replaceOp(origOp, reshapeAfter.output());

    return mlir::success();
}

//
// ScaleShiftOpConverter
//

class ConvertShapeTo4DPass::ScaleShiftOpConverter final : public mlir::OpRewritePattern<IE::ScaleShiftOp> {
public:
    ScaleShiftOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScaleShiftOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertShapeTo4DPass::ScaleShiftOpConverter::matchAndRewrite(
        IE::ScaleShiftOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp.getLoc());

    auto opType = origOp.getType();
    auto newShape = getNewShape(opType);
    if (newShape.empty()) {
        return mlir::failure();
    }

    SmallVector<int64_t> newWeightsShape;
    SmallVector<int64_t> newBiasesShape;
    if (origOp.weights() != nullptr) {
        newWeightsShape = getNewShape(origOp.weights().getType().dyn_cast<mlir::ShapedType>());
        if (newWeightsShape.empty()) {
            return mlir::failure();
        }
    }
    if (origOp.biases() != nullptr) {
        newBiasesShape = getNewShape(origOp.biases().getType().dyn_cast<mlir::ShapedType>());
        if (newBiasesShape.empty()) {
            return mlir::failure();
        }
    }

    mlir::Value newWeights = origOp.weights() != nullptr ? addReshapeOperation(origOp, origOp.weights(),
                                                                               makeArrayRef(newWeightsShape), rewriter)
                                                         : nullptr;
    mlir::Value newBiases = origOp.biases() != nullptr ? addReshapeOperation(origOp, origOp.biases(),
                                                                             makeArrayRef(newBiasesShape), rewriter)
                                                       : nullptr;

    auto reshapeBefore = addReshapeOperation(origOp, origOp.input(), makeArrayRef(newShape), rewriter);

    auto newScaleShiftOp = rewriter.create<IE::ScaleShiftOp>(origOp->getLoc(), reshapeBefore, newWeights, newBiases);

    auto reshapeAfter = addReshapeOperation(origOp, newScaleShiftOp.output(), opType.getShape(), rewriter);
    rewriter.replaceOp(origOp, reshapeAfter.output());

    return mlir::success();
}

//
// MultiplyOpConverter
//

class ConvertShapeTo4DPass::MultiplyOpConverter final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    MultiplyOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertShapeTo4DPass::MultiplyOpConverter::matchAndRewrite(IE::MultiplyOp origOp,
                                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp.getLoc());

    auto opType = origOp.getType();
    auto newInput1Shape = getNewShape(opType);
    auto newInput2Shape = getNewShape(origOp.input2().getType().dyn_cast<mlir::ShapedType>());
    if (newInput1Shape.empty() || newInput2Shape.empty()) {
        return mlir::failure();
    }

    auto reshapedInput1 = addReshapeOperation(origOp, origOp.input1(), makeArrayRef(newInput1Shape), rewriter);
    auto reshapedInput2 = addReshapeOperation(origOp, origOp.input2(), makeArrayRef(newInput2Shape), rewriter);

    auto newMultiplyOp = rewriter.create<IE::MultiplyOp>(origOp->getLoc(), reshapedInput1, reshapedInput2,
                                                         origOp.auto_broadcastAttr());

    auto reshapeAfter = addReshapeOperation(origOp, newMultiplyOp.output(), opType.getShape(), rewriter);
    rewriter.replaceOp(origOp, reshapeAfter.output());

    return mlir::success();
}

//
// AddOpConverter
//

class ConvertShapeTo4DPass::AddOpConverter final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    AddOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertShapeTo4DPass::AddOpConverter::matchAndRewrite(IE::AddOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp.getLoc());

    auto opType = origOp.getType();
    auto newInput1Shape = getNewShape(opType);
    auto newInput2Shape = getNewShape(origOp.input2().getType().dyn_cast<mlir::ShapedType>());
    if (newInput1Shape.empty() || newInput2Shape.empty()) {
        return mlir::failure();
    }

    auto reshapedInput1 = addReshapeOperation(origOp, origOp.input1(), makeArrayRef(newInput1Shape), rewriter);
    auto reshapedInput2 = addReshapeOperation(origOp, origOp.input2(), makeArrayRef(newInput2Shape), rewriter);

    auto newMultiplyOp =
            rewriter.create<IE::AddOp>(origOp->getLoc(), reshapedInput1, reshapedInput2, origOp.auto_broadcastAttr());

    auto reshapeAfter = addReshapeOperation(origOp, newMultiplyOp.output(), opType.getShape(), rewriter);
    rewriter.replaceOp(origOp, reshapeAfter.output());

    return mlir::success();
}

//
// getNewShape
//

SmallVector<int64_t> ConvertShapeTo4DPass::getNewShape(mlir::ShapedType tensor) {
    if (tensor.getShape().size() == TARGET_TENSOR_DIM) {
        return SmallVector<int64_t>();
    } else if (tensor.getShape().size() > TARGET_TENSOR_DIM) {
        VPUX_THROW("Tensors with rank > 4 is not supported");
    } else {
        SmallVector<int64_t> newShape(TARGET_TENSOR_DIM - tensor.getShape().size(), 1);
        newShape.append(tensor.getShape().begin(), tensor.getShape().end());
        return newShape;
    }
}

//
// addReshapeOperation
//

IE::ReshapeOp ConvertShapeTo4DPass::addReshapeOperation(mlir::Operation* origOp, mlir::Value input,
                                                        mlir::ArrayRef<int64_t> shape,
                                                        mlir::PatternRewriter& rewriter) {
    int64_t shapeSize = shape.size();
    const auto inputShapeType = mlir::RankedTensorType::get({shapeSize}, getSInt64Type(origOp->getContext()));
    const auto inputShapeAttr = mlir::DenseElementsAttr::get(inputShapeType, shape);
    auto inputShapeOp = rewriter.create<IE::ConstantOp>(origOp->getLoc(), inputShapeType, inputShapeAttr);
    auto reshapeOp = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), input, inputShapeOp, false);

    return reshapeOp;
}

//
// safeRunOnFunc
//

void ConvertShapeTo4DPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ClampOpConverter>(&ctx, _log);
    patterns.insert<EluOpConverter>(&ctx, _log);
    patterns.insert<ReluOpConverter>(&ctx, _log);
    patterns.insert<SigmoidOpConverter>(&ctx, _log);
    patterns.insert<HSwishOpConverter>(&ctx, _log);
    patterns.insert<TanhOpConverter>(&ctx, _log);
    patterns.insert<FakeQuantizeOpConverter>(&ctx, _log);
    patterns.insert<ScaleShiftOpConverter>(&ctx, _log);
    patterns.insert<MultiplyOpConverter>(&ctx, _log);
    patterns.insert<AddOpConverter>(&ctx, _log);
    IE::ReshapeOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertShapeTo4DPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertShapeTo4DPass(Logger log) {
    return std::make_unique<ConvertShapeTo4DPass>(log);
}

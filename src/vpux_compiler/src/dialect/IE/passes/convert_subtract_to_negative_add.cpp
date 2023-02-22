//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

bool isGreaterShape(ShapeRef shape1, ShapeRef shape2) {
    if (shape1.size() < shape2.size()) {
        return false;
    }
    for (auto p : zip(shape1, shape2)) {
        if (std::get<0>(p) < std::get<1>(p)) {
            return false;
        }
    }
    return true;
}

Const::DeclareOp createShapeConst(mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx, mlir::Location loc,
                                  ShapeRef shape) {
    SmallVector<int64_t> constShape = {4};
    auto intType = getSInt64Type(ctx);
    const auto shapeStorageType = mlir::RankedTensorType::get(constShape, intType);
    const auto shapeDenseAttr = mlir::DenseElementsAttr::get(shapeStorageType, shape.raw());
    auto newContentAttr = Const::ContentAttr::get(shapeDenseAttr).convertElemType(getSInt32Type(ctx));

    return rewriter.create<Const::DeclareOp>(loc, shapeStorageType, newContentAttr);
}

template <typename T>
mlir::DenseElementsAttr negateConstantValues(Const::details::ContentRange<float>& values,
                                             vpux::NDTypeInterface contentType) {
    std::vector<T> negativeValues;
    std::transform(values.begin(), values.end(), std::back_inserter(negativeValues), std::negate<T>());
    return mlir::DenseElementsAttr::get(contentType, makeArrayRef(negativeValues));
}

static inline Const::DeclareOp createConstOpFromValue(mlir::PatternRewriter& rewriter, mlir::Location loc, float val,
                                                      mlir::RankedTensorType argType) {
    const auto denseElementVal = wrapData(argType, val);
    VPUX_THROW_UNLESS(denseElementVal != nullptr,
                      "Subtract pool has incompatible data type {0}, only float16 or float32 are supported",
                      argType.getElementType());

    return rewriter.create<Const::DeclareOp>(loc, argType, Const::ContentAttr::get(denseElementVal));
}

Const::DeclareOp getConstInput(IE::SubtractOp subtractOp) {
    if (auto input2Fq = subtractOp.input2().getDefiningOp<IE::FakeQuantizeOp>()) {
        if (auto fqConstInput = input2Fq.input().getDefiningOp<Const::DeclareOp>()) {
            return fqConstInput;
        }
    } else if (auto input2Const = subtractOp.input2().getDefiningOp<Const::DeclareOp>()) {
        return input2Const;
    }
    return nullptr;
}

mlir::Value createNegativeFqVal(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value fqVal,
                                mlir::RankedTensorType storageType) {
    auto valConst = fqVal.getDefiningOp<Const::DeclareOp>();
    auto valConstContent = valConst.content();

    VPUX_THROW_UNLESS(valConstContent.isSplat(), "Second input's FQ constant is not splat");
    auto inValue = valConstContent.getSplatValue<float>();
    auto negativeValue = (inValue == 0) ? 0 : -1 * inValue;
    return createConstOpFromValue(rewriter, loc, negativeValue, storageType);
}

IE::FakeQuantizeOp createNewFq(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value fqInput,
                               IE::FakeQuantizeOp initialFqOp) {
    auto fqValType = initialFqOp.input_high().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto storageType = mlir::RankedTensorType::get({}, fqValType);

    mlir::Value inLow = createNegativeFqVal(rewriter, loc, initialFqOp.input_high(), storageType);
    mlir::Value inHigh = createNegativeFqVal(rewriter, loc, initialFqOp.input_low(), storageType);
    mlir::Value outLow = createNegativeFqVal(rewriter, loc, initialFqOp.output_high(), storageType);
    mlir::Value outHigh = createNegativeFqVal(rewriter, loc, initialFqOp.output_low(), storageType);

    return rewriter.create<IE::FakeQuantizeOp>(loc, fqInput, inLow, inHigh, outLow, outHigh, initialFqOp.levels(),
                                               initialFqOp.auto_broadcast());
}

//
// ConvertSubtractToDWConvAdd
//

class ConvertSubtractToDWConvAdd final : public mlir::OpRewritePattern<IE::SubtractOp> {
public:
    ConvertSubtractToDWConvAdd(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SubtractOp>(ctx), _log(log) {
        setDebugName("ConvertSubtractToDWConvAdd");
    }

    mlir::LogicalResult matchAndRewrite(IE::SubtractOp subOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertSubtractToDWConvAdd::matchAndRewrite(IE::SubtractOp subOp,
                                                                mlir::PatternRewriter& rewriter) const {
    auto subOpLoc = subOp.getLoc();
    _log.trace("Found SubtractOp at location '{0}'", subOpLoc);

    auto input1 = subOp.input1();
    auto input2 = subOp.input2();

    const auto input1Shape = getShape(input1);
    const auto input2Shape = getShape(input2);
    if (input1Shape.size() != 4 || input2Shape.size() != 4) {
        return mlir::failure();
    }

    const auto elemType = subOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();

    mlir::Value negativeInput = nullptr;

    auto fqInput2 = input2.getDefiningOp<IE::FakeQuantizeOp>();
    auto constInput2 = getConstInput(subOp);
    if (constInput2 != nullptr) {
        auto inputConstContent = constInput2.content();
        const auto contentType = inputConstContent.getType();
        const auto contentElemType = contentType.getElementType();

        const auto rankedTensorType = contentType.cast<mlir::RankedTensorType>();
        auto inValues = inputConstContent.getValues<float>();
        mlir::DenseElementsAttr denseAttr = nullptr;
        if (contentElemType.isF32()) {
            denseAttr = negateConstantValues<float>(inValues, contentType);
        } else if (contentElemType.isF16()) {
            denseAttr = negateConstantValues<float16>(inValues, contentType);
        } else {
            VPUX_THROW("Unsupported type: {0}", contentElemType);
        }
        const auto constContentAttr = Const::ContentAttr::get(denseAttr);

        negativeInput = rewriter.create<Const::DeclareOp>(subOpLoc, rankedTensorType, constContentAttr).output();
    } else {
        const auto inputC = input2Shape[Dims4D::Act::C];
        const Shape filterShape = {inputC, 1, 1, 1};
        const auto filterStorageType = mlir::RankedTensorType::get(to_small_vector(filterShape), elemType);
        auto dwConvFilter = createConstOpFromValue(rewriter, subOpLoc, -1.0f, filterStorageType);
        auto filter = dwConvFilter.output();

        if (fqInput2 != nullptr) {
            const auto fqArgType = mlir::RankedTensorType::get({}, elemType);
            auto fqVal = createConstOpFromValue(rewriter, subOpLoc, -1.0f, fqArgType);
            auto filterFQ = rewriter.create<IE::FakeQuantizeOp>(subOpLoc, filter, fqVal, fqVal, fqVal, fqVal,
                                                                fqInput2.levels(), fqInput2.auto_broadcast());
            filter = filterFQ.output();
        }

        auto dilationsAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
        auto stridesAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
        auto padBeginAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
        auto padEndAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
        auto groupAttr = getIntAttr(rewriter, inputC);

        auto dwConv = rewriter.create<IE::GroupConvolutionOp>(subOpLoc, input2, filter, /*bias=*/nullptr, stridesAttr,
                                                              padBeginAttr, padEndAttr, dilationsAttr, groupAttr,
                                                              /*post_opAttr=*/nullptr);
        negativeInput = dwConv.output();
    }

    if (fqInput2 != nullptr) {
        negativeInput = createNewFq(rewriter, subOpLoc, negativeInput, fqInput2).output();
    }

    if (constInput2 == nullptr) {
        auto negativeInputShape = getShape(negativeInput);
        if (input1Shape != negativeInputShape) {
            auto ctx = rewriter.getContext();

            IE::BroadcastOp inputBroadcast;
            const auto broadcastType = IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY);
            if (isGreaterShape(input1Shape, negativeInputShape)) {
                inputBroadcast = rewriter.create<IE::BroadcastOp>(
                        subOpLoc, negativeInput, createShapeConst(rewriter, ctx, subOpLoc, input1Shape), nullptr,
                        broadcastType);
                negativeInput = inputBroadcast.output();
            } else {
                inputBroadcast = rewriter.create<IE::BroadcastOp>(
                        subOpLoc, input1, createShapeConst(rewriter, ctx, subOpLoc, negativeInputShape), nullptr,
                        broadcastType);
                input1 = inputBroadcast.output();
            }
        }
    }

    rewriter.replaceOpWithNewOp<IE::AddOp>(subOp, input1, negativeInput, subOp.auto_broadcastAttr(),
                                           /*post_op=*/nullptr);
    return mlir::success();
}

//
// ConvertSubtractToNegativeAdd
//

class ConvertSubtractToNegativeAdd final : public mlir::OpRewritePattern<IE::SubtractOp> {
public:
    ConvertSubtractToNegativeAdd(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SubtractOp>(ctx), _log(log) {
        setDebugName("ConvertSubtractToNegativeAdd");
    }

    mlir::LogicalResult matchAndRewrite(IE::SubtractOp subOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertSubtractToNegativeAdd::matchAndRewrite(IE::SubtractOp subOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Found SubtractOp at location '{0}'", subOp.getLoc());

    auto input1 = subOp.input1();
    auto input2 = subOp.input2();

    auto negativeOp = rewriter.create<IE::NegativeOp>(subOp.getLoc(), input2.getType(), input2);

    rewriter.replaceOpWithNewOp<IE::AddOp>(subOp, input1, negativeOp, subOp.auto_broadcastAttr(),
                                           /*post_op=*/nullptr);
    return mlir::success();
}

//
// ConvertSubtractToAddPass
//

class ConvertSubtractToAddPass final : public IE::ConvertSubtractToAddBase<ConvertSubtractToAddPass> {
public:
    explicit ConvertSubtractToAddPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertSubtractToAddPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto module = getOperation();
    const auto arch = VPU::getArch(module);
    mlir::RewritePatternSet patterns(&ctx);
    if (arch == VPU::ArchKind::VPUX30XX) {
        patterns.add<ConvertSubtractToNegativeAdd>(&ctx, _log);
    } else {
        patterns.add<ConvertSubtractToDWConvAdd>(&ctx, _log);
    }

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertSubtractToAddPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertSubtractToAddPass(Logger log) {
    return std::make_unique<ConvertSubtractToAddPass>(log);
}

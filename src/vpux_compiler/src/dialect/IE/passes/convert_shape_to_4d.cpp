//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

constexpr int64_t TARGET_TENSOR_DIM = 4;

//
// ConvertShapeTo4DPass
//

class ConvertShapeTo4DPass final : public IE::ConvertShapeTo4DBase<ConvertShapeTo4DPass> {
public:
    explicit ConvertShapeTo4DPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// GenericConverter
//

mlir::LogicalResult convertGeneric(mlir::Operation* origOp, mlir::ValueRange operands,
                                   mlir::ConversionPatternRewriter& rewriter, mlir::TypeConverter& typeConverter,
                                   Logger log) {
    log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(), "Wrong operands size : {0}", operands.size());

    mlir::BlockAndValueMapping mapper;
    mapper.map(origOperands, operands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto result : newOp->getResults()) {
        result.setType(typeConverter.convertType(result.getType()));
    }

    rewriter.replaceOp(origOp, newOp->getResults());
    return mlir::success();
}

template <class ConcreteOp>
class GenericConverter final : public mlir::OpConversionPattern<ConcreteOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<ConcreteOp>::OpAdaptor;

public:
    GenericConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<ConcreteOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto* typeConverter = this->getTypeConverter();
        VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter was not set");

        return convertGeneric(origOp, newArgs.getOperands(), rewriter, *typeConverter, _log);
    }

private:
    Logger _log;
};

//
// FakeQuantizeConverter
//

class FakeQuantizeConverter final : public mlir::OpConversionPattern<IE::FakeQuantizeOp> {
public:
    FakeQuantizeConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::FakeQuantizeOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

struct ExtendedShape final {
    SmallVector<int64_t> shape;
    Optional<int64_t> quantizeAxis;
};

ExtendedShape extendInputShapeTo4D(IE::FakeQuantizeOp origOp) {
    // If present, axis will be pointing to the channel dimension
    const auto axis = IE::getFQAxisIndex(origOp);
    const auto inShape = origOp.input().getType().cast<vpux::NDTypeInterface>().getShape().raw();

    const auto perAxisCase = [](const auto& inShape, const auto& axis) -> SmallVector<int64_t> {
        // We are trying to place inShape[*axis] dimension to the outShape[1]
        switch (inShape.size()) {
        case 1: {
            return {1, inShape[0], 1, 1};
        }
        case 2: {
            if (*axis == 0) {
                return {1, inShape[0], inShape[1], 1};
            }
            return {inShape[0], inShape[1], 1, 1};
        }
        case 3: {
            if (*axis == 0) {
                return {1, inShape[0], inShape[1], inShape[2]};
            } else if (*axis == 1) {
                return {inShape[0], inShape[1], 1, inShape[2]};
            }
            VPUX_THROW("FakeQuantize 3D case doesn't support axis = {0}", *axis);
        }
        }
        VPUX_THROW("Failed to handle FakeQuantize");
    };

    const auto perTensorCase = [&](const auto& inShape) -> SmallVector<int64_t> {
        auto outShape = to_small_vector(inShape);
        outShape.insert(outShape.begin(), 4 - outShape.size(), 1);
        return outShape;
    };

    const auto outputShape = axis.hasValue() ? perAxisCase(inShape, axis) : perTensorCase(inShape);

    return {outputShape, axis};
}

mlir::Value reshapeConstInput(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value origInput,
                              llvm::Optional<int64_t> axis) {
    const auto inShape = getShape(origInput);
    if (inShape.empty()) {
        return origInput;
    }

    Shape constInputShape{1, 1, 1, 1};
    if (axis.hasValue()) {
        const auto C = Dim(1);
        constInputShape[C] = inShape[Dim(*axis)];
    }

    const auto constInputShapeAttr = getIntArrayAttr(rewriter.getContext(), constInputShape);

    return rewriter.createOrFold<IE::ReshapeOp>(loc, origInput, nullptr, false, constInputShapeAttr);
}

mlir::LogicalResult FakeQuantizeConverter::matchAndRewrite(IE::FakeQuantizeOp origOp, OpAdaptor newArgs,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("[{0}] Found IE::FakeQuantize Operation '{1}'", getDebugName(), origOp->getLoc());

    const auto input4D = extendInputShapeTo4D(origOp);

    const auto inputLow = reshapeConstInput(rewriter, origOp->getLoc(), origOp.input_low(), input4D.quantizeAxis);
    const auto inputHigh = reshapeConstInput(rewriter, origOp->getLoc(), origOp.input_high(), input4D.quantizeAxis);
    const auto outputLow = reshapeConstInput(rewriter, origOp->getLoc(), origOp.output_low(), input4D.quantizeAxis);
    const auto outputHigh = reshapeConstInput(rewriter, origOp->getLoc(), origOp.output_high(), input4D.quantizeAxis);

    const auto newInputShapeAttr = getIntArrayAttr(getContext(), input4D.shape);
    auto inputReshape =
            rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), newArgs.input(), nullptr, false, newInputShapeAttr);

    auto newFakeQuantizeOp =
            rewriter.create<IE::FakeQuantizeOp>(origOp->getLoc(), inputReshape, inputLow, inputHigh, outputLow,
                                                outputHigh, origOp.levels(), origOp.auto_broadcast());

    const auto outputShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.output()));
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newFakeQuantizeOp.output(), nullptr, false, outputShapeAttr);

    _log.trace("[{0}] Replaced with 'IE::FakeQuantize'", getDebugName());

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertShapeTo4DPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto reshape = [](mlir::OpBuilder& builder, mlir::RankedTensorType dstType, mlir::ValueRange inputs,
                            mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());

        const auto outShapeAttr = builder.getI64ArrayAttr(dstType.getShape());
        return builder.createOrFold<IE::ReshapeOp>(loc, inputs.front(), nullptr, false, outShapeAttr);
    };

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](vpux::NDTypeInterface type) {
        if (type.getRank() == TARGET_TENSOR_DIM) {
            return type;
        } else if (type.getRank() > TARGET_TENSOR_DIM) {
            VPUX_THROW("Tensors with rank > 4 are not supported");
        } else if (type.getRank() == 3) {
            SmallVector<int64_t> newShape = {type.getShape()[Dim(0)], type.getShape()[Dim(1)], 1,
                                             type.getShape()[Dim(2)]};
            return type.changeShape(ShapeRef(newShape));
        } else if (type.getRank() == 2) {
            SmallVector<int64_t> newShape = {1, type.getShape()[Dim(0)], 1, type.getShape()[Dim(1)]};
            return type.changeShape(ShapeRef(newShape));
        } else {
            SmallVector<int64_t> newShape(TARGET_TENSOR_DIM - type.getRank(), 1);
            newShape.append(type.getShape().begin(), type.getShape().end());
            return type.changeShape(ShapeRef(newShape));
        }
    });
    typeConverter.addSourceMaterialization(reshape);
    typeConverter.addTargetMaterialization(reshape);
    typeConverter.addArgumentMaterialization(reshape);

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    const auto isLegalFqOp = [&](IE::FakeQuantizeOp op) {
        const auto inShape = op.input().getType().cast<vpux::NDTypeInterface>().getShape();
        const auto outShape = op.output().getType().cast<vpux::NDTypeInterface>().getShape();

        VPUX_THROW_WHEN(inShape != outShape,
                        "FakeQuantize must have the same shape for input and output. Got: {0} != {1}", inShape,
                        outShape);
        VPUX_THROW_WHEN(inShape.size() > TARGET_TENSOR_DIM, "Tensors with rank > 4 are not supported");

        return inShape.size() == TARGET_TENSOR_DIM;
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<IE::IEDialect>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::FuncOp>();
    target.addLegalOp<mlir::ReturnOp>();
    target.addDynamicallyLegalOp<IE::ClampOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::EluOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReLUOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SigmoidOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::HSwishOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::TanhOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SinOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::CosOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SqrtOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SinhOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::CoshOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AsinhOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AcoshOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AtanhOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ExpOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AddOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::MultiplyOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SubtractOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AndOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ScaleShiftOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::EqualOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::NotEqualOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::FakeQuantizeOp>(isLegalFqOp);
    target.addDynamicallyLegalOp<IE::LessOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LessEqualOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GreaterOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GreaterEqualOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LogicalNotOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LogicalOrOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LogicalXorOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AbsOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AtanOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AsinOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AcosOp>(isLegalOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericConverter<IE::ClampOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::EluOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::ReLUOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SigmoidOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::HSwishOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::TanhOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SinOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::CosOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SqrtOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SinhOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::CoshOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AsinhOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AcoshOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AtanhOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::ExpOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AddOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::MultiplyOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::SubtractOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AndOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::ScaleShiftOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::EqualOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LessOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LessEqualOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::NotEqualOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::GreaterOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::GreaterEqualOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LogicalNotOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LogicalOrOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::LogicalXorOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AbsOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AtanOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AsinOp>>(typeConverter, &ctx, _log);
    patterns.add<GenericConverter<IE::AcosOp>>(typeConverter, &ctx, _log);
    patterns.add<FakeQuantizeConverter>(typeConverter, &ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
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

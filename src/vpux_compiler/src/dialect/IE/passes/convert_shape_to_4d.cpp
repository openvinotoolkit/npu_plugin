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
    void safeRunOnModule() final;
};

//
// GenericConverter
//

mlir::LogicalResult convertGeneric(mlir::Operation* origOp, ArrayRef<mlir::Value> operands,
                                   mlir::ConversionPatternRewriter& rewriter, mlir::TypeConverter& typeConverter,
                                   Logger log) {
    log.trace("Process Operation '{0}'", origOp->getLoc());

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
public:
    GenericConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<ConcreteOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto* typeConverter = this->getTypeConverter();
        VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter was not set");

        return convertGeneric(origOp, operands, rewriter, *typeConverter, _log);
    }

private:
    Logger _log;
};

//
// FakeQuantizeConverter
//

struct ExtendedShape {
    SmallVector<int64_t> shape;
    llvm::Optional<int64_t> quantizeAxis;
};

class FakeQuantizeConverter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    FakeQuantizeConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;
    static ExtendedShape extendInputShapeTo4D(IE::FakeQuantizeOp origOp);
    static mlir::Value reshapeConstInput(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                                         llvm::Optional<int64_t> axis);
    static llvm::Optional<int64_t> getAxisIndex(IE::FakeQuantizeOp fq);

private:
    Logger _log;
};

ExtendedShape FakeQuantizeConverter::extendInputShapeTo4D(IE::FakeQuantizeOp origOp) {
    const auto inShape = origOp.input().getType().cast<mlir::ShapedType>().getShape();
    VPUX_THROW_UNLESS(inShape.size() < 4, "Tensors with rank >= 4 could not be converted to 4D");

    // If present, axis will be pointing to the channel dimension
    const auto axis = getAxisIndex(origOp);

    const auto perAxisCase = [](const auto& inShape, const auto& axis) -> SmallVector<int64_t> {
        // We are trying to place inShape[*axis] dimension to the outShape[1]
        switch (inShape.size()) {
        case 1: {
            return {1, inShape[0], 1, 1};
        }
        case 2: {
            VPUX_THROW_UNLESS(*axis == 1, "FakeQuantize constant input has incorrect shape");
            return {inShape[0], inShape[1], 1, 1};
        }
        case 3: {
            if (*axis == 0) {
                return {1, inShape[0], inShape[1], inShape[2]};
            } else if (*axis == 1) {
                return {inShape[0], inShape[1], 1, inShape[2]};
            }
            VPUX_THROW("FakeQuantize 3D case don't support axis = {0}", *axis);
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

llvm::Optional<int64_t> FakeQuantizeConverter::getAxisIndex(IE::FakeQuantizeOp fq) {
    const auto extractAxis = [](mlir::Value input) -> llvm::Optional<int64_t> {
        const auto greaterThanOne = [](auto dim) {
            return dim > 1;
        };
        const auto shape = getShape(input);

        const auto axisCount = llvm::count_if(shape, greaterThanOne);
        VPUX_THROW_UNLESS(axisCount <= 1, "FakeQuantize constant input has incorrect shape");

        auto axis = llvm::find_if(shape, greaterThanOne);
        if (axis != shape.end()) {
            return std::distance(shape.begin(), axis);
        }

        return llvm::Optional<int64_t>{};
    };

    const auto inputLowAxis = extractAxis(fq.input_low());
    const auto outputLowAxis = extractAxis(fq.output_low());

    if (!inputLowAxis && !outputLowAxis) {
        return {};
    }

    if (inputLowAxis && outputLowAxis) {
        VPUX_THROW_UNLESS(*inputLowAxis == *outputLowAxis, "FakeQuantize constant inputs use different axis");
    }

    return inputLowAxis ? *inputLowAxis : *outputLowAxis;
}

mlir::Value FakeQuantizeConverter::reshapeConstInput(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                                     mlir::Value input, llvm::Optional<int64_t> axis) {
    VPUX_THROW_UNLESS(input, "FakeQuantize input is null");

    const auto inShape = getShape(input);
    const auto fqPerTensor = inShape.empty();
    if (fqPerTensor) {
        return input;
    }

    auto constInputShape = SmallVector<int64_t>{1, 1, 1, 1};
    if (axis.hasValue()) {
        constInputShape[*axis] = inShape[Dim(*axis)];
    }

    const auto constInputShapeAttr = getIntArrayAttr(rewriter.getContext(), constInputShape);

    return rewriter.create<IE::ReshapeOp>(loc, input, nullptr, false, constInputShapeAttr);
}

mlir::LogicalResult FakeQuantizeConverter::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::FakeQuantize Operation '{0}'", origOp->getLoc());

    if (getShape(origOp.input()).size() >= 4) {
        return mlir::failure();
    }

    const auto input4D = extendInputShapeTo4D(origOp);

    const auto inputLow = reshapeConstInput(rewriter, origOp->getLoc(), origOp.input_low(), input4D.quantizeAxis);
    const auto inputHigh = reshapeConstInput(rewriter, origOp->getLoc(), origOp.input_high(), input4D.quantizeAxis);
    const auto outputLow = reshapeConstInput(rewriter, origOp->getLoc(), origOp.output_low(), input4D.quantizeAxis);
    const auto outputHigh = reshapeConstInput(rewriter, origOp->getLoc(), origOp.output_high(), input4D.quantizeAxis);

    const auto newInputShapeAttr = getIntArrayAttr(getContext(), input4D.shape);
    auto inputReshape =
            rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.input(), nullptr, false, newInputShapeAttr);

    auto newFakeQuantizeOp =
            rewriter.create<IE::FakeQuantizeOp>(origOp->getLoc(), inputReshape.output(), inputLow, inputHigh, outputLow,
                                                outputHigh, origOp.levels(), origOp.auto_broadcast());

    const auto outputShapeAttr = getIntArrayAttr(getContext(), getShape(origOp.output()));
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newFakeQuantizeOp.output(), nullptr, false, outputShapeAttr);

    _log.trace("Replaced with 'IE::FakeQuantize'");

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertShapeTo4DPass::safeRunOnModule() {
    auto& ctx = getContext();

    const auto reshape = [](mlir::OpBuilder& builder, mlir::RankedTensorType dstType, mlir::ValueRange inputs,
                            mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());

        const auto outShapeAttr = builder.getI64ArrayAttr(dstType.getShape());
        return builder.createOrFold<IE::ReshapeOp>(loc, inputs.front(), nullptr, false, outShapeAttr);
    };

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::RankedTensorType tensor) {
        if (tensor.getRank() == TARGET_TENSOR_DIM) {
            return tensor;
        } else if (tensor.getRank() > TARGET_TENSOR_DIM) {
            VPUX_THROW("Tensors with rank > 4 are not supported");
        } else {
            SmallVector<int64_t> newShape(TARGET_TENSOR_DIM - tensor.getRank(), 1);
            newShape.append(tensor.getShape().begin(), tensor.getShape().end());
            return changeShape(tensor, ShapeRef(newShape));
        }
    });
    typeConverter.addSourceMaterialization(reshape);
    typeConverter.addTargetMaterialization(reshape);
    typeConverter.addArgumentMaterialization(reshape);

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<IE::IEDialect>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getType());
    });
    target.addDynamicallyLegalOp<mlir::ReturnOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::FakeQuantizeOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ClampOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::EluOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReLUOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SigmoidOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::HSwishOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::TanhOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ExpOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AddOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::MultiplyOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ScaleShiftOp>(isLegalOp);

    mlir::RewritePatternSet patterns(&ctx);
    mlir::populateFuncOpTypeConversionPattern(patterns, typeConverter);
    patterns.insert<GenericConverter<mlir::ReturnOp>>(typeConverter, &ctx, _log);
    patterns.insert<GenericConverter<IE::ClampOp>>(typeConverter, &ctx, _log);
    patterns.insert<GenericConverter<IE::EluOp>>(typeConverter, &ctx, _log);
    patterns.insert<GenericConverter<IE::ReLUOp>>(typeConverter, &ctx, _log);
    patterns.insert<GenericConverter<IE::SigmoidOp>>(typeConverter, &ctx, _log);
    patterns.insert<GenericConverter<IE::HSwishOp>>(typeConverter, &ctx, _log);
    patterns.insert<GenericConverter<IE::TanhOp>>(typeConverter, &ctx, _log);
    patterns.insert<GenericConverter<IE::ExpOp>>(typeConverter, &ctx, _log);
    patterns.insert<GenericConverter<IE::AddOp>>(typeConverter, &ctx, _log);
    patterns.insert<GenericConverter<IE::MultiplyOp>>(typeConverter, &ctx, _log);
    patterns.insert<GenericConverter<IE::ScaleShiftOp>>(typeConverter, &ctx, _log);
    patterns.insert<FakeQuantizeConverter>(&ctx, _log);

    auto module = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
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

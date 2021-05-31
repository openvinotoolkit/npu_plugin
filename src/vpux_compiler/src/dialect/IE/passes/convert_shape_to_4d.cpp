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
            VPUX_THROW("Tensors with rank > 4 is not supported");
        } else {
            SmallVector<int64_t> newShape(TARGET_TENSOR_DIM - tensor.getRank(), 1);
            newShape.append(tensor.getShape().begin(), tensor.getShape().end());
            return mlir::RankedTensorType::get(newShape, tensor.getElementType());
        }
    });
    typeConverter.addSourceMaterialization(reshape);
    typeConverter.addTargetMaterialization(reshape);
    typeConverter.addArgumentMaterialization(reshape);

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
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
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::ConstantOp>();

    mlir::RewritePatternSet patterns(&ctx);
    mlir::populateFuncOpTypeConversionPattern(patterns, typeConverter);
    patterns.insert<GenericConverter<mlir::ReturnOp>>(typeConverter, &ctx, _log);
    patterns.insert<GenericConverter<IE::FakeQuantizeOp>>(typeConverter, &ctx, _log);
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

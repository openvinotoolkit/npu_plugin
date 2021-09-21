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
#include "vpux/compiler/dialect/IE/utils/generic_op_converter.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertPrecisionToFP16Pass
//

class ConvertPrecisionToFP16Pass final : public IE::ConvertPrecisionToFP16Base<ConvertPrecisionToFP16Pass> {
public:
    explicit ConvertPrecisionToFP16Pass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertPrecisionToFP16Pass::safeRunOnModule() {
    auto& ctx = getContext();

    const auto cvtType = [](mlir::OpBuilder& builder, mlir::RankedTensorType type, mlir::ValueRange inputs,
                            mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        return builder.createOrFold<IE::ConvertOp>(loc, inputs[0], mlir::TypeAttr::get(type.getElementType()));
    };

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::RankedTensorType tensor) {
        if (tensor.getElementType().isF32()) {
            return changeElemType(tensor, mlir::Float16Type::get(tensor.getContext()));
        } else {
            return tensor;
        }
    });
    typeConverter.addSourceMaterialization(cvtType);
    typeConverter.addTargetMaterialization(cvtType);

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addDynamicallyLegalDialect<IE::IEDialect>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::ReturnOp>(isLegalOp);
    target.addLegalOp<IE::ConvertOp>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getType());
    });

    mlir::RewritePatternSet patterns(&ctx);
    mlir::populateFuncOpTypeConversionPattern(patterns, typeConverter);
    patterns.insert<GenericOpConverter>(typeConverter, &ctx, _log);
    IE::ConvertOp::getCanonicalizationPatterns(patterns, &ctx);

    auto module = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertPrecisionToFP16Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPrecisionToFP16Pass(Logger log) {
    return std::make_unique<ConvertPrecisionToFP16Pass>(log);
}

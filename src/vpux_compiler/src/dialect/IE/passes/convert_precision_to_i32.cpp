//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/convert_op_types.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;
using namespace IE;

namespace {

//
// ConvertPrecisionToI32Pass
//

class ConvertPrecisionToI32Pass final : public IE::ConvertPrecisionToI32Base<ConvertPrecisionToI32Pass> {
public:
    explicit ConvertPrecisionToI32Pass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertPrecisionToI32Pass::safeRunOnModule() {
    auto& ctx = getContext();

    mlir::TypeConverter typeConverter;
    setupConvertPrecision(typeConverter, [](mlir::Type elemType) -> mlir::Type {
        if (elemType.isSignedInteger(64)) {
            return mlir::IntegerType::get(elemType.getContext(), 32, mlir::IntegerType::Signed);
        } else {
            return elemType;
        }
    });

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addDynamicallyLegalDialect<IE::IEDialect>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GatherOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::BroadcastOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::RollOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceMaxOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceMeanOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceSumOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceProdOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceMinOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceL1Op>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceL2Op>(isLegalOp);
    target.addDynamicallyLegalOp<IE::TopKOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AdaptiveAvgPoolOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AdaptiveMaxPoolOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::ReturnOp>(isLegalOp);
    target.addLegalOp<mlir::ModuleOp>();
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getType());
    });

    // Convert TopK and AdaptiveMaxPool element type attribute to avoid failures in infer return type checking.
    auto module = getOperation();
    module.walk([&](IE::TopKOp op) {
        mlir::Type sInt32Type = mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        op->setAttr("element_type", mlir::TypeAttr::get(sInt32Type));
    });
    module.walk([&](IE::AdaptiveMaxPoolOp op) {
        mlir::Type sInt32Type = mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        op->setAttr("index_element_type", mlir::TypeAttr::get(sInt32Type));
    });
    if (mlir::failed(runConvertPrecision(module, typeConverter, target, _log))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertPrecisionToI32Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPrecisionToI32Pass(Logger log) {
    return std::make_unique<ConvertPrecisionToI32Pass>(log);
}

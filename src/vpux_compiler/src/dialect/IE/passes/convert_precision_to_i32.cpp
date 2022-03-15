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
    target.addDynamicallyLegalOp<IE::ReduceMaxOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceMeanOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceSumOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::TopKOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::ReturnOp>(isLegalOp);
    target.addLegalOp<mlir::ModuleOp>();
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getType());
    });

    // Convert TopK element type attribute to avoid failures in infer return type checking.
    auto module = getOperation();
    module.walk([&](IE::TopKOp op) {
        mlir::Type sInt32Type = mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        op->setAttr("element_type", mlir::TypeAttr::get(sInt32Type));
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

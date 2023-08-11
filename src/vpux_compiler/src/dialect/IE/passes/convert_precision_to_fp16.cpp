//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/convert_op_types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;
using namespace IE;

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

    mlir::TypeConverter typeConverter;
    setupConvertPrecision(typeConverter, [](mlir::Type elemType) -> mlir::Type {
        if (elemType.isF32() || elemType.isSignlessInteger(CHAR_BIT)) {
            return mlir::Float16Type::get(elemType.getContext());
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
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::OneHotOp>(isLegalOp);
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<IE::DynamicQuantizeOp>();
    // AssignOp & ReadValueOp represent inputs/outputs. Cannot convert their type internally.
    target.addLegalOp<IE::AssignOp>();
    target.addLegalOp<IE::ReadValueOp>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getFunctionType());
    });

    auto module = getOperation();

    // For output element type is inferred based on an attribute
    auto adjustOpAttrPrecisions = [&](IE::OneHotOp op) {
        if (op.outElemTypeAttr() == mlir::TypeAttr::get(mlir::Float32Type::get(&ctx))) {
            auto outElemType = mlir::Float16Type::get(&ctx);
            op.outElemTypeAttr(mlir::TypeAttr::get(outElemType));
        }
    };
    module.walk(adjustOpAttrPrecisions);

    module.walk([&](IE::RandomUniformOp op) {
        if (op.output_typeAttr() == mlir::TypeAttr::get(mlir::Float32Type::get(&ctx))) {
            auto outElemType = mlir::Float16Type::get(&ctx);
            op.output_typeAttr(mlir::TypeAttr::get(outElemType));
        }
    });

    if (mlir::failed(runConvertPrecision(module, typeConverter, target, _log))) {
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

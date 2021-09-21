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
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

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

    const auto cvtType = [](mlir::OpBuilder& builder, mlir::RankedTensorType type, mlir::ValueRange inputs,
                            mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        return builder.createOrFold<IE::ConvertOp>(loc, inputs[0], mlir::TypeAttr::get(type.getElementType()));
    };

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::RankedTensorType tensor) {
        if (tensor.getElementType().isSignedInteger(64)) {
            return changeElemType(tensor, mlir::IntegerType::get(tensor.getContext(), 32, mlir::IntegerType::Signed));
        } else {
            return tensor;
        }
    });
    typeConverter.addSourceMaterialization(cvtType);
    typeConverter.addTargetMaterialization(cvtType);

    const auto isLegalGatherOp = [&](IE::GatherOp op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addDynamicallyLegalOp<IE::GatherOp>(isLegalGatherOp);
    target.addLegalOp<IE::ConvertOp>();

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
// createConvertPrecisionToI32Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPrecisionToI32Pass(Logger log) {
    return std::make_unique<ConvertPrecisionToI32Pass>(log);
}

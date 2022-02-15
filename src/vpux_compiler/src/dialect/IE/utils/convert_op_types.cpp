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

#include "vpux/compiler/dialect/IE/utils/convert_op_types.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;
using namespace IE;

namespace {

//
// ConvertOpTypes
//

class ConvertOpTypes final : public mlir::ConversionPattern {
public:
    ConvertOpTypes(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, vpux::Logger log)
            : mlir::ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, vpux::benefitHigh, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, vpux::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    vpux::Logger _log;
};

mlir::LogicalResult ConvertOpTypes::matchAndRewrite(mlir::Operation* origOp, vpux::ArrayRef<mlir::Value> operands,
                                                    mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp->getLoc());

    auto* converter = getTypeConverter();
    VPUX_THROW_UNLESS(converter != nullptr, "TypeConverter was not set");

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(), "Wrong operands size : {0}", operands.size());

    mlir::BlockAndValueMapping mapper;
    mapper.map(origOperands, operands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto result : newOp->getResults()) {
        result.setType(converter->convertType(result.getType()));
    }

    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

}  // namespace

void vpux::IE::setupConvertPrecision(mlir::TypeConverter& typeConverter,
                                     FuncRef<mlir::Type(mlir::Type)> elemTypeConversionCb) {
    typeConverter.addConversion([elemTypeConversionCb](vpux::NDTypeInterface tensor) {
        return tensor.changeElemType(elemTypeConversionCb(tensor.getElementType()));
    });

    const auto convert = [](mlir::OpBuilder& builder, mlir::RankedTensorType type, mlir::ValueRange inputs,
                            mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        return builder.createOrFold<IE::ConvertOp>(loc, inputs[0], mlir::TypeAttr::get(type.getElementType()));
    };

    typeConverter.addSourceMaterialization(convert);
    typeConverter.addTargetMaterialization(convert);
    typeConverter.addArgumentMaterialization(convert);
}

mlir::LogicalResult vpux::IE::runConvertPrecision(mlir::ModuleOp module, mlir::TypeConverter& typeConverter,
                                                  mlir::ConversionTarget& target, Logger& log) {
    target.addLegalOp<IE::ConvertOp>();

    mlir::RewritePatternSet patterns(module.getContext());
    mlir::populateFuncOpTypeConversionPattern(patterns, typeConverter);
    patterns.add<ConvertOpTypes>(typeConverter, module.getContext(), log);

    return mlir::applyPartialConversion(module, target, std::move(patterns));
}

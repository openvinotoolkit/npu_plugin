//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/mlir/attributes.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

namespace {

class AdjustPrecisionForVPUPass final
        : public IE::AdjustPrecisionForVPUBase<AdjustPrecisionForVPUPass> {
public:
    AdjustPrecisionForVPUPass();

public:
    void runOnOperation() final;

public:
    class FuncOpConverter;
    class OpConverter;

private:
    void passBody();

private:
    mlir::OpPassManager _cleanUpIR;
};

AdjustPrecisionForVPUPass::AdjustPrecisionForVPUPass()
        : _cleanUpIR(mlir::ModuleOp::getOperationName(),
                     mlir::OpPassManager::Nesting::Implicit) {
    _cleanUpIR.addPass(mlir::createCanonicalizerPass());
}

//
// FuncOpConverter
//

class AdjustPrecisionForVPUPass::FuncOpConverter final
        : public mlir::OpConversionPattern<mlir::FuncOp> {
public:
    using mlir::OpConversionPattern<mlir::FuncOp>::OpConversionPattern;

public:
    mlir::LogicalResult matchAndRewrite(
            mlir::FuncOp funcOp,
            ArrayRef<mlir::Value> operands,
            mlir::ConversionPatternRewriter& rewriter) const final;
};

mlir::LogicalResult AdjustPrecisionForVPUPass::FuncOpConverter::matchAndRewrite(
        mlir::FuncOp funcOp,
        ArrayRef<mlir::Value>,
        mlir::ConversionPatternRewriter& rewriter) const {
    auto* converter = getTypeConverter();
    VPUX_THROW_UNLESS(converter != nullptr, "TypeConverter was not set");

    const auto funcType = funcOp.getType();

    mlir::TypeConverter::SignatureConversion conversion(
            funcType.getNumInputs());
    for (const auto& p : funcType.getInputs() | indexed) {
        const auto newType = converter->convertType(p.value());
        conversion.addInputs(checked_cast<uint32_t>(p.index()), newType);
    }

    SmallVector<mlir::Type, 1> newResultTypes;
    newResultTypes.reserve(funcOp.getNumResults());
    for (const auto& outType : funcType.getResults()) {
        newResultTypes.push_back(converter->convertType(outType));
    }

    if (mlir::failed(rewriter.convertRegionTypes(&funcOp.getBody(),
                                                 *converter,
                                                 &conversion))) {
        return printTo(funcOp.emitError(),
                       "Failed to convert Function arguments");
    }

    rewriter.updateRootInPlace(funcOp, [&]() {
        funcOp.setType(rewriter.getFunctionType(conversion.getConvertedTypes(),
                                                newResultTypes));
    });

    return mlir::success();
}

//
// OpConverter
//

class AdjustPrecisionForVPUPass::OpConverter final
        : public mlir::ConversionPattern {
public:
    OpConverter(mlir::TypeConverter& typeConverter,
                mlir::PatternBenefit benefit = 1)
            : mlir::ConversionPattern(benefit,
                                      typeConverter,
                                      MatchAnyOpTypeTag{}) {
    }

public:
    mlir::LogicalResult matchAndRewrite(
            mlir::Operation* origOp,
            ArrayRef<mlir::Value> operands,
            mlir::ConversionPatternRewriter& rewriter) const final;
};

mlir::LogicalResult AdjustPrecisionForVPUPass::OpConverter::matchAndRewrite(
        mlir::Operation* origOp,
        ArrayRef<mlir::Value> operands,
        mlir::ConversionPatternRewriter& rewriter) const {
    auto* converter = getTypeConverter();
    VPUX_THROW_UNLESS(converter != nullptr, "TypeConverter was not set");

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(),
                      "Wrong operands size : {0}",
                      operands.size());

    mlir::BlockAndValueMapping mapper;
    mapper.map(origOperands, operands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto result : newOp->getResults()) {
        result.setType(converter->convertType(result.getType()));
    }

    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// IEConvertToFP16Pass::runOnOperation
//

void AdjustPrecisionForVPUPass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        printTo(getOperation().emitError(),
                "AdjustPrecisionForVPUPass failed : {0}",
                e.what());
        signalPassFailure();
    }
}

void AdjustPrecisionForVPUPass::passBody() {
    auto& ctx = getContext();

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::RankedTensorType tensor) {
        if (tensor.getElementType().isF32()) {
            return mlir::RankedTensorType::get(
                    tensor.getShape(),
                    mlir::Float16Type::get(tensor.getContext()));
        } else if (tensor.getElementType().isIntOrIndex()) {
            return mlir::RankedTensorType::get(
                    tensor.getShape(),
                    getSInt32Type(tensor.getContext()));
        } else {
            return tensor;
        }
    });
    typeConverter.addSourceMaterialization(
            [](mlir::OpBuilder& builder,
               mlir::RankedTensorType type,
               mlir::ValueRange inputs,
               mlir::Location loc) -> mlir::Value {
                VPUX_THROW_UNLESS(inputs.size() == 1,
                                  "Got wrong number of inputs : {0}",
                                  inputs.size());
                return builder.createOrFold<IE::ConvertOp>(
                        loc,
                        inputs[0],
                        mlir::TypeAttr::get(type.getElementType()));
            });
    typeConverter.addTargetMaterialization(
            [](mlir::OpBuilder& builder,
               mlir::RankedTensorType type,
               mlir::ValueRange inputs,
               mlir::Location loc) -> mlir::Value {
                VPUX_THROW_UNLESS(inputs.size() == 1,
                                  "Got wrong number of inputs : {0}",
                                  inputs.size());
                return builder.createOrFold<IE::ConvertOp>(
                        loc,
                        inputs[0],
                        mlir::TypeAttr::get(type.getElementType()));
            });

    const auto isLegalOp = [&](mlir::Operation* op) {
        return llvm::all_of(op->getOperandTypes(),
                            [&](mlir::Type type) {
                                return typeConverter.isLegal(type);
                            }) &&
               llvm::all_of(op->getResultTypes(), [&](mlir::Type type) {
                   return typeConverter.isLegal(type);
               });
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalDialect<IE::IEDialect>(isLegalOp);
    target.addLegalOp<IE::ConvertOp>();
    target.addIllegalDialect<mlir::StandardOpsDialect>();
    target.addDynamicallyLegalOp<mlir::ReturnOp>(isLegalOp);
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getType()) &&
               typeConverter.isLegal(&funcOp.getBody());
    });

    mlir::OwningRewritePatternList patterns;
    patterns.insert<FuncOpConverter>(typeConverter, &ctx);
    patterns.insert<OpConverter>(typeConverter);

    auto module = getOperation();
    if (mlir::failed(mlir::applyFullConversion(module.getOperation(),
                                               target,
                                               std::move(patterns)))) {
        signalPassFailure();
    }

    if (mlir::failed(runPipeline(_cleanUpIR, module))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustPrecisionForVPUPass() {
    return std::make_unique<AdjustPrecisionForVPUPass>();
}

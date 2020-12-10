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
#include "vpux/compiler/utils/scalars.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#define DEBUG_TYPE "convert_nd_ops_to_4d"
#define TARGET_TENSOR_DIM 4

using namespace vpux;

namespace {

class ConvertNDOpsTo4DPass final : public IE::ConvertNDOpsTo4DBase<ConvertNDOpsTo4DPass> {
public:
    explicit ConvertNDOpsTo4DPass(Logger log);

public:
    void runOnOperation() final;

public:
    class FuncOpConverter;
    class ConstantOpConverter;
    class GenericOpConverter;

private:
    void passBody();

private:
    Logger _log;
    mlir::OpPassManager _cleanUpIR;
};

ConvertNDOpsTo4DPass::ConvertNDOpsTo4DPass(Logger log)
        : _log(log), _cleanUpIR(mlir::ModuleOp::getOperationName(), mlir::OpPassManager::Nesting::Implicit) {
    _log.setName(Base::getArgumentName());

    _cleanUpIR.addPass(mlir::createCanonicalizerPass());
}

void ConvertNDOpsTo4DPass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        printTo(getOperation().emitError(), "ConvertNDOpsTo4DPass failed : {0}", e.what());
        signalPassFailure();
    }
}

//
// FuncOpConverter
//

class ConvertNDOpsTo4DPass::FuncOpConverter final : public mlir::OpConversionPattern<mlir::FuncOp> {
public:
    using mlir::OpConversionPattern<mlir::FuncOp>::OpConversionPattern;

public:
    mlir::LogicalResult matchAndRewrite(mlir::FuncOp funcOp, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertNDOpsTo4DPass::FuncOpConverter::matchAndRewrite(
        mlir::FuncOp funcOp, ArrayRef<mlir::Value>, mlir::ConversionPatternRewriter& rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "ConvertNDOpsTo4DPass::FuncOpConverter\n");
    LLVM_DEBUG({
        llvm::dbgs() << "- Orig FuncOp:  ";
        funcOp.dump();
    });

    auto* converter = getTypeConverter();
    VPUX_THROW_UNLESS(converter != nullptr, "TypeConverter was not set");

    const auto funcType = funcOp.getType();

    mlir::TypeConverter::SignatureConversion conversion(funcType.getNumInputs());
    for (const auto& p : funcType.getInputs() | indexed) {
        const auto newType = converter->convertType(p.value());
        conversion.addInputs(checked_cast<uint32_t>(p.index()), newType);
    }

    SmallVector<mlir::Type, 1> newResultTypes;
    newResultTypes.reserve(funcOp.getNumResults());
    for (const auto& outType : funcType.getResults()) {
        newResultTypes.push_back(converter->convertType(outType));
    }

    if (mlir::failed(rewriter.convertRegionTypes(&funcOp.getBody(), *converter, &conversion))) {
        return printTo(funcOp.emitError(), "Failed to convert Function arguments");
    }

    rewriter.updateRootInPlace(funcOp, [&]() {
        funcOp.setType(rewriter.getFunctionType(conversion.getConvertedTypes(), newResultTypes));
    });
    LLVM_DEBUG({
        llvm::dbgs() << "- New FuncOp:  ";
        funcOp.dump();
    });
    return mlir::success();
}

//
// ConstantOpConverter
//

class ConvertNDOpsTo4DPass::ConstantOpConverter final : public mlir::OpConversionPattern<mlir::ConstantOp> {
public:
    ConstantOpConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context,
                        mlir::PatternBenefit benefit = 2)
            : mlir::OpConversionPattern<mlir::ConstantOp>(typeConverter, context, benefit) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::ConstantOp origOp, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertNDOpsTo4DPass::ConstantOpConverter::matchAndRewrite(
        mlir::ConstantOp origOp, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "ConvertNDOpsTo4DPass::ConstantOpConverter\n");
    LLVM_DEBUG({
        llvm::dbgs() << "- Orig ConstantOp:  ";
        origOp.dump();
    });

    auto* converter = getTypeConverter();
    VPUX_THROW_UNLESS(converter != nullptr, "TypeConverter was not set");

    VPUX_THROW_UNLESS(operands.empty(), "Wrong operands size : {0}", operands.size());

    auto origTensorType = origOp.getResult().getType().dyn_cast<mlir::RankedTensorType>();
    if (origTensorType == nullptr || origTensorType.getShape().size() == TARGET_TENSOR_DIM) {
        return mlir::failure();
    }

    auto origContent = origOp.value().dyn_cast<mlir::DenseElementsAttr>();
    if (origContent == nullptr) {
        return mlir::failure();
    }

    auto newType = converter->convertType(origTensorType).cast<mlir::ShapedType>();

    mlir::DenseElementsAttr newContent = origContent.reshape(newType);

    auto* dialect = rewriter.getContext()->getLoadedDialect<IE::IEDialect>();
    VPUX_THROW_UNLESS(dialect != nullptr, "Got NULL pointer for IEDialect");

    auto* newOp = dialect->materializeConstant(rewriter, newContent, newType, origOp.getLoc());
    rewriter.replaceOp(origOp, newOp->getResults());

    LLVM_DEBUG({
        llvm::dbgs() << "- New ConstantOp:  ";
        newOp->dump();
    });

    return mlir::success();
}

//
// GenericOpConverter
//

class ConvertNDOpsTo4DPass::GenericOpConverter final : public mlir::ConversionPattern {
public:
    GenericOpConverter(mlir::TypeConverter& shapeConverter, mlir::PatternBenefit benefit = 1)
            : mlir::ConversionPattern(benefit, shapeConverter, MatchAnyOpTypeTag{}) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertNDOpsTo4DPass::GenericOpConverter::matchAndRewrite(
        mlir::Operation* origOp, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "ConvertNDOpsTo4DPass::GenericConverter\n");
    LLVM_DEBUG({
        llvm::dbgs() << "- Orig OP:  ";
        origOp->dump();
    });
    auto* converter = getTypeConverter();
    VPUX_THROW_UNLESS(converter != nullptr, "TypeConverter was not set");

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(), "Wrong operands size : {0}", operands.size());

    mlir::BlockAndValueMapping mapper;
    mapper.map(origOperands, operands);

    if (auto softMaxOp = mlir::dyn_cast<IE::SoftMaxOp>(*origOp)) {
        // TODO: implement this within op as an interface?
        auto newAxis = softMaxOp.axisInd() +
                       (TARGET_TENSOR_DIM -
                        origOp->getOperand(0).getType().dyn_cast<mlir::RankedTensorType>().getShape().size());
        softMaxOp.axisIndAttr(rewriter.getI32IntegerAttr(newAxis));
    }

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto result : newOp->getResults()) {
        result.setType(converter->convertType(result.getType()));
    }
    rewriter.replaceOp(origOp, newOp->getResults());

    LLVM_DEBUG({
        llvm::dbgs() << "- New OP:  ";
        newOp->dump();
    });

    return mlir::success();
}

void ConvertNDOpsTo4DPass::passBody() {
    auto& ctx = getContext();

    mlir::TypeConverter shapeConverter;
    shapeConverter.addConversion([](mlir::RankedTensorType tensor) {
        if (tensor.getShape().size() == TARGET_TENSOR_DIM)
            return tensor;
        else if (tensor.getShape().size() > TARGET_TENSOR_DIM)
            VPUX_THROW("# dims > 4 is not supporeted");
        else {
            int32_t nDimsToAdd = TARGET_TENSOR_DIM - (int32_t)tensor.getShape().size();
            SmallVector<int64_t, TARGET_TENSOR_DIM> newShape(nDimsToAdd, 1);
            for (auto& s : tensor.getShape()) {
                newShape.push_back(s);
            }
            return mlir::RankedTensorType::get(newShape, tensor.getElementType());
        }
    });

    shapeConverter.addSourceMaterialization([](mlir::OpBuilder& builder, mlir::RankedTensorType type,
                                               mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        return builder.create<mlir::linalg::TensorReshapeOp>(loc, type, inputs[0]);
    });
    shapeConverter.addTargetMaterialization([](mlir::OpBuilder& builder, mlir::RankedTensorType type,
                                               mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        return builder.create<mlir::linalg::TensorReshapeOp>(loc, type, inputs[0]);
    });

    const auto isLegalOp = [&](mlir::Operation* op) {
        return shapeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addDynamicallyLegalDialect<IE::IEDialect>(isLegalOp);
    target.addDynamicallyLegalDialect<mlir::StandardOpsDialect>(isLegalOp);

    target.addLegalOp<IE::ConvertOp>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

    target.addDynamicallyLegalOp<mlir::ConstantOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::ReturnOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
        return shapeConverter.isSignatureLegal(funcOp.getType()) && shapeConverter.isLegal(&funcOp.getBody());
    });

    mlir::OwningRewritePatternList patterns;
    patterns.insert<GenericOpConverter>(shapeConverter);
    patterns.insert<FuncOpConverter>(shapeConverter, &ctx);
    patterns.insert<ConstantOpConverter>(shapeConverter, &ctx);

    auto module = getOperation();

    if (mlir::failed(mlir::applyFullConversion(module.getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }

    if (mlir::failed(runPipeline(_cleanUpIR, module))) {
        signalPassFailure();
    }
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createConvertNDOpsTo4DPass(Logger log) {
    return std::make_unique<ConvertNDOpsTo4DPass>(log);
}

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
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

namespace {

constexpr size_t TARGET_TENSOR_DIM = 4;

//
// ConvertShapeTo4DPass
//

class ConvertShapeTo4DPass final : public IE::ConvertShapeTo4DBase<ConvertShapeTo4DPass> {
public:
    explicit ConvertShapeTo4DPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    void runOnOperation() final;

public:
    class GenericOpConverter;

private:
    void passBody();

private:
    Logger _log;
};

void ConvertShapeTo4DPass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        (void)errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// GenericOpConverter
//

class ConvertShapeTo4DPass::GenericOpConverter final : public mlir::ConversionPattern {
public:
    GenericOpConverter(mlir::TypeConverter& shapeConverter, Logger log)
            : mlir::ConversionPattern(1 /*benefit*/, shapeConverter, MatchAnyOpTypeTag{}), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertShapeTo4DPass::GenericOpConverter::matchAndRewrite(
        mlir::Operation* origOp, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp->getLoc());

    auto* converter = getTypeConverter();
    VPUX_THROW_UNLESS(converter != nullptr, "TypeConverter was not set");

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(), "Wrong operands size : {0}", operands.size());

    mlir::BlockAndValueMapping mapper;
    mapper.map(origOperands, operands);

    // TODO: implement this within op as an interface?
    if (auto softMaxOp = mlir::dyn_cast<IE::SoftMaxOp>(*origOp)) {
        auto newAxis = softMaxOp.axisInd() +
                       (TARGET_TENSOR_DIM -
                        origOp->getOperand(0).getType().dyn_cast<mlir::RankedTensorType>().getShape().size());
        softMaxOp.axisIndAttr(rewriter.getI32IntegerAttr(checked_cast<int32_t>(newAxis)));
    }

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto result : newOp->getResults()) {
        result.setType(converter->convertType(result.getType()));
    }
    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// passBody
//

void ConvertShapeTo4DPass::passBody() {
    auto& ctx = getContext();

    const auto cvtType = [](mlir::OpBuilder& builder, mlir::RankedTensorType type, mlir::ValueRange inputs,
                            mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        return builder.createOrFold<mlir::linalg::TensorReshapeOp>(loc, type, inputs[0]);
    };

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::RankedTensorType tensor) {
        if (tensor.getShape().size() == TARGET_TENSOR_DIM) {
            return tensor;
        } else if (tensor.getShape().size() > TARGET_TENSOR_DIM) {
            VPUX_THROW("Tensors with rank > 4 is not supported");
        } else {
            const auto nDimsToAdd = TARGET_TENSOR_DIM - tensor.getShape().size();
            SmallVector<int64_t> newShape(nDimsToAdd, 1);
            for (auto s : tensor.getShape()) {
                newShape.push_back(s);
            }
            return mlir::RankedTensorType::get(newShape, tensor.getElementType());
        }
    });
    typeConverter.addSourceMaterialization(cvtType);
    typeConverter.addTargetMaterialization(cvtType);

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalDialect<IE::IEDialect>(isLegalOp);
    target.addLegalOp<mlir::linalg::TensorReshapeOp>();
    target.addDynamicallyLegalOp<mlir::ReturnOp>(isLegalOp);
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getType()) && typeConverter.isLegal(&funcOp.getBody());
    });

    mlir::OwningRewritePatternList patterns(&ctx);
    mlir::populateFuncOpTypeConversionPattern(patterns, typeConverter);
    patterns.insert<GenericOpConverter>(typeConverter, _log.nest());
    mlir::linalg::TensorReshapeOp::getCanonicalizationPatterns(patterns, &ctx);

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

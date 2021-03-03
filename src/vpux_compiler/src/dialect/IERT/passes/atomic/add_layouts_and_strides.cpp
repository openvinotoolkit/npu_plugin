//
// Copyright 2021 Intel Corporation.
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

#include <mlir/Transforms/DialectConversion.h>
#include <vpux/compiler/core/attributes/stride_reqs.hpp>
#include "vpux/compiler/dialect/IERT/passes.hpp"

using namespace vpux;

namespace {

//
// AddLayoutsAndStridesPass
//

class AddLayoutsAndStridesPass final : public IERT::AddLayoutsAndStridesBase<AddLayoutsAndStridesPass> {
public:
    explicit AddLayoutsAndStridesPass(Logger log): _log(log) {
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

void AddLayoutsAndStridesPass::runOnOperation() {
    try {
        passBody();
    } catch (const std::exception& e) {
        errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// GenericOpConverter
//

class AddLayoutsAndStridesPass::GenericOpConverter final : public mlir::ConversionPattern {
public:
    GenericOpConverter(mlir::TypeConverter& typeConverter, Logger log)
            : mlir::ConversionPattern(1 /*benefit*/, typeConverter, MatchAnyOpTypeTag{}), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AddLayoutsAndStridesPass::GenericOpConverter::matchAndRewrite(
        mlir::Operation* origOp, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp->getLoc());

    auto* converter = getTypeConverter();
    VPUX_THROW_UNLESS(converter != nullptr, "TypeConverter was not set");

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(), "Wrong operands size : {0}", operands.size());

    mlir::BlockAndValueMapping mapper;
    mapper.map(origOperands, operands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto ops : newOp->getResults()) {
        ops.setType(converter->convertType(ops.getType()));
    }

    if (origOp->getNumResults()) {
        rewriter.replaceOp(origOp, newOp->getResults());
    } else {
        rewriter.eraseOp(origOp);
    }

    return mlir::success();
}

//
// passBody
//

void AddLayoutsAndStridesPass::passBody() {
    auto& ctx = getContext();

    const auto cvtType = [](mlir::OpBuilder& builder, mlir::MemRefType type, mlir::ValueRange inputs,
                            mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        SmallVector<mlir::Value> results{};
        builder.createOrFold<mlir::UnrealizedConversionCastOp>(results, loc, type, inputs);
        VPUX_THROW_UNLESS(results.size() == 1, "Got wrong number of outputs : {0}", results.size());
        return results.front();
    };

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::MemRefType type) {
        const auto affineMap = type.getAffineMaps();
        if (!affineMap.empty()) {
            return type;
        }

        const auto elemSize = getElemTypeSize(type);
        const auto strides =
                to_small_vector(StrideReqs::simple().calcStrides(type) | reversed | transformed([&](Bit val) {
                                    return val.count() / elemSize.count();
                                }));

        const auto order = DimsOrder::fromNumDims(type.getRank());
        const auto stridesMap = mlir::makeStridedLinearLayoutMap(strides, 0, type.getContext());
        return mlir::MemRefType::get(type.getShape(), type.getElementType(),
                                     {order.toAffineMap(type.getContext()), stridesMap}, type.getMemorySpace());
    });
    typeConverter.addSourceMaterialization(cvtType);
    typeConverter.addTargetMaterialization(cvtType);

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalDialect<IERT::IERTDialect>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::ReturnOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::linalg::ReshapeOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::linalg::CopyOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::AllocOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::DeallocOp>(isLegalOp);
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    target.addLegalOp<IE::CNNNetworkOp>();
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getType());
    });

    mlir::OwningRewritePatternList patterns;
    mlir::populateFuncOpTypeConversionPattern(patterns, &ctx, typeConverter);
    patterns.insert<GenericOpConverter>(typeConverter, _log.nest());

    auto module = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAddLayoutsAndStridesPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createAddLayoutsAndStridesPass(Logger log) {
    return std::make_unique<AddLayoutsAndStridesPass>(log);
}

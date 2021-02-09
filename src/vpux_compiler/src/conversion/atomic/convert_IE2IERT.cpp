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

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Transforms/Bufferize.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// ConvertIE2IERTPass
//

class ConvertIE2IERTPass final : public ConvertIE2IERTBase<ConvertIE2IERTPass> {
public:
    explicit ConvertIE2IERTPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    void runOnFunction() final;

public:
    class ConstantRewrite;
    class QuantRewrite;
    class LayerRewrite;

public:
    static const mlir::PatternBenefit genericBenefit;
    static const mlir::PatternBenefit specificBenefit;

public:
    static SmallVector<mlir::Value> allocateResults(mlir::Location loc, mlir::OpBuilder& builder,
                                                    mlir::TypeConverter& typeConverter, mlir::ValueRange origResults);

private:
    void passBody();

private:
    Logger _log;
};

const mlir::PatternBenefit ConvertIE2IERTPass::genericBenefit(1);
const mlir::PatternBenefit ConvertIE2IERTPass::specificBenefit(2);

void ConvertIE2IERTPass::runOnFunction() {
    try {
        _log.trace("Run on Function '@{0}'", getFunction().sym_name());

        passBody();
    } catch (const std::exception& e) {
        errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// allocateResults
//

SmallVector<mlir::Value> ConvertIE2IERTPass::allocateResults(mlir::Location loc, mlir::OpBuilder& builder,
                                                             mlir::TypeConverter& typeConverter,
                                                             mlir::ValueRange origResults) {
    return to_small_vector(origResults | transformed([&](mlir::Value origVal) -> mlir::Value {
                               auto origType = origVal.getType();
                               auto memRefType = typeConverter.convertType(origType);
                               auto allocOp = builder.create<mlir::AllocOp>(loc, memRefType.cast<mlir::MemRefType>());
                               return allocOp.memref();
                           }));
}

//
// ConstantRewrite
//

class ConvertIE2IERTPass::ConstantRewrite final : public mlir::OpConversionPattern<IE::ConstantOp> {
public:
    ConstantRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ConstantOp>(typeConverter, ctx, specificBenefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConstantOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIE2IERTPass::ConstantRewrite::matchAndRewrite(
        IE::ConstantOp origOp, ArrayRef<mlir::Value>, mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Constant Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newType = typeConverter->convertType(origOp.getType());

    _log.trace("Create IERT analogue");
    rewriter.replaceOpWithNewOp<IERT::ConstantOp>(origOp, newType, origOp.value());

    return mlir::success();
}

//
// QuantRewrite
//

class ConvertIE2IERTPass::QuantRewrite final : public mlir::ConversionPattern {
public:
    QuantRewrite(mlir::TypeConverter& typeConverter, Logger log)
            : mlir::ConversionPattern(specificBenefit, typeConverter, MatchAnyOpTypeTag{}), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIE2IERTPass::QuantRewrite::matchAndRewrite(mlir::Operation* origOp,
                                                                      ArrayRef<mlir::Value> newOperands,
                                                                      mlir::ConversionPatternRewriter& rewriter) const {
    if (origOp->getDialect()->getTypeID() != mlir::TypeID::get<mlir::quant::QuantizationDialect>()) {
        return mlir::failure();
    }

    _log.trace("Found Quant Operation '{0}'", origOp->getLoc());

    if (origOp->getNumOperands() != 1 || origOp->getNumResults() != 1) {
        _log.trace("Unsupported number of inputs/outputs");
        return mlir::failure();
    }

    if (!mlir::isa<mlir::quant::QuantizeCastOp, mlir::quant::DequantizeCastOp>(origOp)) {
        _log.trace("Unsupported Operation type '{0}'", origOp->getName());
        return mlir::failure();
    }

    VPUX_THROW_UNLESS(newOperands.size() == 1, "Got wrong newOperands size : '{0}'", newOperands.size());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    _log.trace("Add Alloc Operations for results");
    const auto allocatedBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, {origOp->getOpResult(0)});

    _log.trace("Create IERT analogue");
    llvm::TypeSwitch<mlir::Operation*, void>(origOp)
            .Case<mlir::quant::QuantizeCastOp>([&](mlir::quant::QuantizeCastOp origOp) {
                rewriter.create<IERT::QuantizeOp>(origOp.getLoc(), newOperands[0], allocatedBufs[0]);
            })
            .Case<mlir::quant::DequantizeCastOp>([&](mlir::quant::DequantizeCastOp origOp) {
                rewriter.create<IERT::DequantizeOp>(origOp.getLoc(), newOperands[0], allocatedBufs[0]);
            });

    rewriter.replaceOp(origOp, allocatedBufs);

    return mlir::success();
}

//
// LayerRewrite
//

class ConvertIE2IERTPass::LayerRewrite final : public mlir::ConversionPattern {
public:
    LayerRewrite(mlir::TypeConverter& typeConverter, Logger log)
            : mlir::ConversionPattern(genericBenefit, typeConverter, mlir::Pattern::MatchAnyOpTypeTag{}), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIE2IERTPass::LayerRewrite::matchAndRewrite(mlir::Operation* origOp,
                                                                      ArrayRef<mlir::Value> newOperands,
                                                                      mlir::ConversionPatternRewriter& rewriter) const {
    auto layerOp = mlir::dyn_cast<LayerInterface>(origOp);
    if (layerOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Found Layer Operation '{0}'", origOp->getLoc());

    auto origInputs = layerOp.getInputs();
    auto origOutputs = layerOp.getOutputs();
    VPUX_THROW_UNLESS(newOperands.size() == origInputs.size(), "Got wrong newOperands size : '{0}', expected '{1}'",
                      newOperands.size(), origInputs.size());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    _log.trace("Add Alloc Operations for results");
    auto allocatedBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOutputs);

    const auto newOpName =
            llvm::formatv("{0}.{1}", IERT::IERTDialect::getDialectNamespace(), origOp->getName().stripDialect()).str();
    _log.trace("Create IERT analogue : '{0}'", newOpName);

    mlir::OperationState newOpDef(origOp->getLoc(), newOpName);
    newOpDef.addOperands(newOperands);
    newOpDef.addOperands(allocatedBufs);
    newOpDef.addAttributes(origOp->getAttrs());

    rewriter.createOperation(newOpDef);
    rewriter.replaceOp(origOp, allocatedBufs);

    return mlir::success();
}

//
// passBody
//

void ConvertIE2IERTPass::passBody() {
    auto& ctx = getContext();

    mlir::BufferizeTypeConverter typeConverter;

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<IERT::IERTDialect>();
    target.addIllegalDialect<IE::IEDialect>();
    target.addIllegalDialect<mlir::quant::QuantizationDialect>();
    target.addLegalOp<IE::CNNNetworkOp, IE::DataInfoOp, IE::EndOp>();
    target.addLegalOp<mlir::AllocOp>();
    mlir::populateBufferizeMaterializationLegality(target);

    mlir::OwningRewritePatternList patterns;
    patterns.insert<ConstantRewrite>(typeConverter, &ctx, _log.nest());
    patterns.insert<QuantRewrite>(typeConverter, _log.nest());
    patterns.insert<LayerRewrite>(typeConverter, _log.nest());

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertIE2IERTPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertIE2IERTPass(Logger log) {
    return std::make_unique<ConvertIE2IERTPass>(log);
}

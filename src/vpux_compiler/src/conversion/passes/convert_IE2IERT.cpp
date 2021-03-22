//
// Copyright Intel Corporation.
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
#include "vpux/compiler/utils/attributes.hpp"

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
    class LinalgReshapeRewrite;
    class GenericReshapeRewrite;
    class LayerRewrite;
    class DetectionOutputRewrite;
    class ScaleShiftRewrite;
    class TransposeRewrite;

public:
    static const mlir::PatternBenefit genericBenefit;
    static const mlir::PatternBenefit specificBenefitLow;
    static const mlir::PatternBenefit specificBenefitHigh;

public:
    static SmallVector<mlir::Value> allocateResults(mlir::Location loc, mlir::OpBuilder& builder,
                                                    mlir::TypeConverter& typeConverter, mlir::ValueRange origResults);

private:
    void passBody();

private:
    Logger _log;
};

const mlir::PatternBenefit ConvertIE2IERTPass::genericBenefit(1);
const mlir::PatternBenefit ConvertIE2IERTPass::specificBenefitLow(2);
const mlir::PatternBenefit ConvertIE2IERTPass::specificBenefitHigh(3);

void ConvertIE2IERTPass::runOnFunction() {
    try {
        _log.trace("Run on Function '@{0}'", getFunction().sym_name());

        passBody();
    } catch (const std::exception& e) {
        (void)errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
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
                               auto allocOp =
                                       builder.create<mlir::memref::AllocOp>(loc, memRefType.cast<mlir::MemRefType>());
                               return allocOp.memref();
                           }));
}

//
// ConstantRewrite
//

class ConvertIE2IERTPass::ConstantRewrite final : public mlir::OpConversionPattern<IE::ConstantOp> {
public:
    ConstantRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ConstantOp>(typeConverter, ctx, specificBenefitHigh), _log(log) {
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
            : mlir::ConversionPattern(specificBenefitHigh, typeConverter, MatchAnyOpTypeTag{}), _log(log) {
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
// LinalgReshapeRewrite
//

class ConvertIE2IERTPass::LinalgReshapeRewrite final : public mlir::OpConversionPattern<mlir::linalg::TensorReshapeOp> {
public:
    LinalgReshapeRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<mlir::linalg::TensorReshapeOp>(typeConverter, ctx, specificBenefitHigh),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::linalg::TensorReshapeOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIE2IERTPass::LinalgReshapeRewrite::matchAndRewrite(
        mlir::linalg::TensorReshapeOp origOp, ArrayRef<mlir::Value> newOperands,
        mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found TensorReshape Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    VPUX_THROW_UNLESS(newOperands.size() == 1, "Got wrong newOperands size : '{0}'", newOperands.size());

    const auto newType = typeConverter->convertType(origOp.getType());

    rewriter.replaceOpWithNewOp<mlir::linalg::ReshapeOp>(origOp, newType, newOperands[0], origOp.reassociation());

    return mlir::success();
}

//
// ReshapeRewrite
//

class ConvertIE2IERTPass::GenericReshapeRewrite final : public mlir::ConversionPattern {
public:
    GenericReshapeRewrite(mlir::TypeConverter& typeConverter, Logger log)
            : mlir::ConversionPattern(specificBenefitLow, typeConverter, MatchAnyOpTypeTag{}), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIE2IERTPass::GenericReshapeRewrite::matchAndRewrite(
        mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands, mlir::ConversionPatternRewriter& rewriter) const {
    if (!mlir::isa<mlir::ViewLikeOpInterface>(origOp)) {
        return mlir::failure();
    }
    if (origOp->getNumResults() != 1) {
        return mlir::failure();
    }

    _log.trace("Found ViewLike Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto outType = origOp->getResult(0).getType().cast<mlir::ShapedType>();

    if (!outType.hasStaticShape()) {
        _log.nest().trace("Dynamic shapes are not supported yet");
        return mlir::failure();
    }

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(outType);

    VPUX_THROW_UNLESS(!newOperands.empty(), "Got wrong newOperands size : '{0}'", newOperands.size());

    rewriter.replaceOpWithNewOp<IERT::GenericReshapeOp>(origOp, newOutType, newOperands[0]);

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
// DetectionOutputRewrite
//

class ConvertIE2IERTPass::DetectionOutputRewrite final : public mlir::OpConversionPattern<IE::DetectionOutputOp> {
public:
    DetectionOutputRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::DetectionOutputOp>(typeConverter, ctx, specificBenefitHigh), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DetectionOutputOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIE2IERTPass::DetectionOutputRewrite::matchAndRewrite(
        IE::DetectionOutputOp origOp, ArrayRef<mlir::Value> newOperands,
        mlir::ConversionPatternRewriter& rewriter) const {
    auto origInputs = origOp.getInputs();
    auto origOutputs = origOp.getOutputs();
    VPUX_THROW_UNLESS(newOperands.size() == origInputs.size(), "Got wrong newOperands size : '{0}', expected '{1}'",
                      newOperands.size(), origInputs.size());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    _log.trace("Add Alloc Operations for results");
    auto allocatedBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOutputs);

    mlir::Value inAdditionalPreds = nullptr;
    mlir::Value inAdditionalProposals = nullptr;
    if (newOperands.size() == 5) {
        inAdditionalPreds = newOperands[3];
        inAdditionalProposals = newOperands[4];
    }

    _log.trace("Create an IERT analog of the IE::DetectionOutput");
    rewriter.create<IERT::DetectionOutputOp>(origOp->getLoc(), newOperands[0], newOperands[1], newOperands[2],
                                             inAdditionalPreds, inAdditionalProposals, allocatedBufs[0], origOp.attr());

    rewriter.replaceOp(origOp, allocatedBufs);
    return mlir::success();
}

//
// ScaleShiftRewrite
//

class ConvertIE2IERTPass::ScaleShiftRewrite final : public mlir::OpConversionPattern<IE::ScaleShiftOp> {
public:
    ScaleShiftRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ScaleShiftOp>(typeConverter, ctx, specificBenefitHigh), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIE2IERTPass::ScaleShiftRewrite::matchAndRewrite(
        IE::ScaleShiftOp origOp, ArrayRef<mlir::Value> newOperands, mlir::ConversionPatternRewriter& rewriter) const {
    auto origInputs = origOp.getInputs();
    auto origOutputs = origOp.getOutputs();
    VPUX_THROW_UNLESS(newOperands.size() == origInputs.size(), "Got wrong newOperands size : '{0}', expected '{1}'",
                      newOperands.size(), origInputs.size());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    _log.trace("Add Alloc Operations for results");
    auto allocatedBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOutputs);

    mlir::Value newWeights = nullptr;
    mlir::Value newBiases = nullptr;
    if (origOp.weights() != nullptr && origOp.biases() != nullptr) {
        newWeights = newOperands[1];
        newBiases = newOperands[2];
    } else if (origOp.weights() != nullptr) {
        newWeights = newOperands[1];
    } else if (origOp.biases() != nullptr) {
        newBiases = newOperands[1];
    } else {
        VPUX_THROW("ScaleShift must have weights or biases");
    }

    _log.trace("Create an IERT analog of the IE::ScaleShift");
    rewriter.create<IERT::ScaleShiftOp>(origOp->getLoc(), newOperands[0], newWeights, newBiases, allocatedBufs[0]);

    rewriter.replaceOp(origOp, allocatedBufs);
    return mlir::success();
}

//
// TransposeRewrite
//

class ConvertIE2IERTPass::TransposeRewrite final : public mlir::OpConversionPattern<IE::TransposeOp> {
public:
    TransposeRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::TransposeOp>(typeConverter, ctx, specificBenefitHigh), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIE2IERTPass::TransposeRewrite::matchAndRewrite(
        IE::TransposeOp origOp, ArrayRef<mlir::Value> newOperands, mlir::ConversionPatternRewriter& rewriter) const {
    const auto origInputs = origOp.getInputs();
    const auto origOutputs = origOp.getOutputs();
    VPUX_THROW_UNLESS(newOperands.size() == origInputs.size(), "Got wrong newOperands size : '{0}', expected '{1}'",
                      newOperands.size(), origInputs.size());

    if (!origOp.order_value().hasValue()) {
        return errorAt(origOp, "Order attribute is empty. This case should be handled by canonicalizer of TransposeOp");
    }

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    _log.trace("Add Alloc Operations for results");
    auto allocatedBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOutputs);

    _log.trace("Create an IERT analog of the IE::TransposeOp");
    rewriter.create<IERT::TransposeOp>(origOp->getLoc(), newOperands[0], allocatedBufs[0],
                                       origOp.order_value().getValue());

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
    target.addIllegalOp<mlir::linalg::TensorReshapeOp>();
    target.addLegalOp<IE::CNNNetworkOp, IE::DataInfoOp, IE::EndOp>();
    target.addLegalOp<mlir::memref::AllocOp>();
    target.addLegalOp<mlir::linalg::ReshapeOp>();
    mlir::populateBufferizeMaterializationLegality(target);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<DetectionOutputRewrite>(typeConverter, &ctx, _log.nest());
    patterns.insert<ScaleShiftRewrite>(typeConverter, &ctx, _log.nest());
    patterns.insert<TransposeRewrite>(typeConverter, &ctx, _log.nest());
    patterns.insert<ConstantRewrite>(typeConverter, &ctx, _log.nest());
    patterns.insert<QuantRewrite>(typeConverter, _log.nest());
    patterns.insert<LinalgReshapeRewrite>(typeConverter, &ctx, _log.nest());
    patterns.insert<GenericReshapeRewrite>(typeConverter, _log.nest());
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

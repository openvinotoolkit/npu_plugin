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

#include "vpux/utils/core/range.hpp"

#include <mlir/Transforms/Bufferize.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// BufferizeIEPass
//

class BufferizeIEPass final : public BufferizeIEBase<BufferizeIEPass> {
public:
    explicit BufferizeIEPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    void runOnFunction() final;

public:
    class LayerRewrite;

public:
    static const mlir::PatternBenefit genericBenefit;
    static const mlir::PatternBenefit specificBenefit;

public:
    static SmallVector<mlir::Value, 1> allocateResults(mlir::Location loc, mlir::OpBuilder& builder,
                                                       mlir::TypeConverter& typeConverter,
                                                       mlir::ValueRange origResults);

private:
    void passBody();

private:
    Logger _log;
};

const mlir::PatternBenefit BufferizeIEPass::genericBenefit(1);
const mlir::PatternBenefit BufferizeIEPass::specificBenefit(2);

void BufferizeIEPass::runOnFunction() {
    try {
        _log.trace("Run on Function '@{0}'", getFunction().sym_name());

        passBody();
    } catch (const std::exception& e) {
        printTo(getOperation().emitError(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// allocateResults
//

SmallVector<mlir::Value, 1> BufferizeIEPass::allocateResults(mlir::Location loc, mlir::OpBuilder& builder,
                                                             mlir::TypeConverter& typeConverter,
                                                             mlir::ValueRange origResults) {
    return to_vector<1>(origResults | transformed([&](mlir::Value origVal) -> mlir::Value {
                            auto origType = origVal.getType();
                            auto memRefType = typeConverter.convertType(origType);
                            auto allocOp = builder.create<mlir::AllocOp>(loc, memRefType.cast<mlir::MemRefType>());
                            return allocOp.memref();
                        }));
}

//
// LayerRewrite
//

class BufferizeIEPass::LayerRewrite final : public mlir::ConversionPattern {
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

mlir::LogicalResult BufferizeIEPass::LayerRewrite::matchAndRewrite(mlir::Operation* origOp,
                                                                   ArrayRef<mlir::Value> newOperands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
    auto layerOp = mlir::dyn_cast<LayerInterface>(origOp);
    if (layerOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Found Layer Operation '{0}'", *origOp);

    auto origInputs = layerOp.getInputs();
    auto origOutputs = layerOp.getOutputs();
    VPUX_THROW_UNLESS(newOperands.size() == origInputs.size(), "Got wrong newOperands size : '{0}', expected '{1}'",
                      newOperands.size(), origInputs.size());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    _log.trace("Add Alloc Operations for results");
    auto allocatedBufs = allocateResults(origOp->getLoc(), rewriter, *typeConverter, origOutputs);

    llvm::TypeSwitch<mlir::Operation*, void>(origOp)
            .Case<IE::ConvertOp>([&](IE::ConvertOp) {
                rewriter.create<IERT::ConvertOp>(origOp->getLoc(), newOperands[0], allocatedBufs[0]);
            })
            .Case<IE::SoftMaxOp>([&](IE::SoftMaxOp op) {
                rewriter.create<IERT::SoftMaxOp>(origOp->getLoc(), newOperands[0], allocatedBufs[0], op.axisIndAttr());
            })
            .Default([](mlir::Operation* op) {
                VPUX_THROW("Got unsupported layer Operation '{0}'", op->getName());
            });

    rewriter.replaceOp(origOp, allocatedBufs);

    _log.trace("Replaced with IERT analogue");

    return mlir::success();
}

//
// passBody
//

void BufferizeIEPass::passBody() {
    auto& ctx = getContext();

    mlir::BufferizeTypeConverter typeConverter;

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<IERT::IERTDialect>();
    target.addIllegalDialect<IE::IEDialect>();
    target.addLegalOp<IE::CNNNetworkOp, IE::DataInfoOp, IE::EndOp>();
    target.addLegalOp<mlir::AllocOp>();
    target.addDynamicallyLegalOp<mlir::ConstantOp>([&](mlir::ConstantOp op) {
        return typeConverter.isLegal(op);
    });
    mlir::populateBufferizeMaterializationLegality(target);

    mlir::OwningRewritePatternList patterns;
    patterns.insert<LayerRewrite>(typeConverter, _log.nest());

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createBufferizeIEPass
//

std::unique_ptr<mlir::Pass> vpux::createBufferizeIEPass(Logger log) {
    return std::make_unique<BufferizeIEPass>(log);
}

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

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/utils/core/func_ref.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// SetInternalMemorySpacePass
//

class SetInternalMemorySpacePass final : public IERT::SetInternalMemorySpaceBase<SetInternalMemorySpacePass> {
public:
    SetInternalMemorySpacePass(mlir::Attribute memSpace, Logger log);

public:
    void runOnFunction() final;

public:
    class AllocRewrite;
    class GenericRewrite;

public:
    static const mlir::PatternBenefit genericBenefit;
    static const mlir::PatternBenefit specificBenefit;

private:
    void passBody();

private:
    mlir::Attribute _memSpace;
    Logger _log;
};

const mlir::PatternBenefit SetInternalMemorySpacePass::genericBenefit(1);
const mlir::PatternBenefit SetInternalMemorySpacePass::specificBenefit(2);

SetInternalMemorySpacePass::SetInternalMemorySpacePass(mlir::Attribute memSpace, Logger log)
        : _memSpace(memSpace), _log(log) {
    _log.setName(Base::getArgumentName());
}

void SetInternalMemorySpacePass::runOnFunction() {
    try {
        auto& ctx = getContext();

        if (_memSpace == nullptr) {
            VPUX_THROW_UNLESS(!memSpaceName.getValue().empty(), "Missing memory space option");
            _memSpace = mlir::StringAttr::get(memSpaceName.getValue(), &ctx);
        }

        passBody();
    } catch (const std::exception& e) {
        printTo(getOperation().emitError(), "{0} failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// AllocRewrite
//

class SetInternalMemorySpacePass::AllocRewrite final : public mlir::OpConversionPattern<mlir::AllocOp> {
public:
    AllocRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<mlir::AllocOp>(typeConverter, ctx, specificBenefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::AllocOp origOp, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SetInternalMemorySpacePass::AllocRewrite::matchAndRewrite(
        mlir::AllocOp origOp, ArrayRef<mlir::Value>, mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}'", origOp);

    auto* converter = getTypeConverter();
    VPUX_THROW_UNLESS(converter != nullptr, "TypeConverter was not set");

    const auto val = origOp.memref();

    const auto origType = val.getType().dyn_cast<mlir::MemRefType>();
    if (origType == nullptr) {
        _log.trace("Got unsupported type '{0}'", val.getType());
        return mlir::failure();
    }

    const auto newType = converter->convertType(origType).cast<mlir::MemRefType>();
    rewriter.replaceOpWithNewOp<mlir::AllocOp>(origOp, newType);

    _log.trace("Replaced with new Alloc Operation with type '{0}'", newType);

    return mlir::success();
}

//
// GenericRewrite
//

class SetInternalMemorySpacePass::GenericRewrite final : public mlir::ConversionPattern {
public:
    explicit GenericRewrite(Logger log)
            : mlir::ConversionPattern(genericBenefit, mlir::ConversionPattern::MatchAnyOpTypeTag()), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SetInternalMemorySpacePass::GenericRewrite::matchAndRewrite(
        mlir::Operation* origOp, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", *origOp);

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(), "Wrong operands size : {0}", operands.size());

    mlir::BlockAndValueMapping mapper;
    mapper.map(origOperands, operands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// passBody
//

void SetInternalMemorySpacePass::passBody() {
    auto& ctx = getContext();

    using MaterializeCallback = FuncRef<mlir::Value(mlir::OpBuilder & builder, mlir::MemRefType type,
                                                    mlir::ValueRange inputs, mlir::Location loc)>;

    const auto materializeBuffer = [this](mlir::OpBuilder& builder, mlir::MemRefType type, mlir::ValueRange inputs,
                                          mlir::Location loc) -> mlir::Value {
        auto cvtLog = _log.nest();
        cvtLog.trace("Materialize buffer for '{0}'", type);

        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());

        auto allocOp = builder.create<mlir::AllocOp>(loc, type);
        builder.create<mlir::linalg::CopyOp>(loc, inputs[0], allocOp.memref());

        return allocOp.memref();
    };

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([this](mlir::MemRefType mem) {
        return mlir::MemRefType::get(mem.getShape(), mem.getElementType(), mem.getAffineMaps(), _memSpace);
    });
    typeConverter.addArgumentMaterialization(MaterializeCallback(materializeBuffer));
    typeConverter.addSourceMaterialization(MaterializeCallback(materializeBuffer));
    typeConverter.addTargetMaterialization(MaterializeCallback(materializeBuffer));

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalDialect<IERT::IERTDialect>([&](mlir::Operation* op) {
        for (auto arg : op->getOperands()) {
            if (auto allocProducer = arg.getDefiningOp<mlir::AllocOp>()) {
                if (!typeConverter.isLegal(allocProducer)) {
                    return false;
                }
            }
        }
        return true;
    });
    target.addDynamicallyLegalOp<mlir::AllocOp>([&](mlir::AllocOp op) {
        return typeConverter.isLegal(op);
    });
    target.addLegalOp<mlir::linalg::CopyOp>();
    target.addLegalOp<mlir::ConstantOp>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

    mlir::OwningRewritePatternList patterns;
    patterns.insert<AllocRewrite>(typeConverter, &ctx, _log.nest());
    patterns.insert<GenericRewrite>(_log.nest());

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IERT::createSetInternalMemorySpacePass(mlir::Attribute memSpace, Logger log) {
    return std::make_unique<SetInternalMemorySpacePass>(memSpace, log);
}

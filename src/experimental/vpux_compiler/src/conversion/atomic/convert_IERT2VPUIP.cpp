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

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// Generated
//

#include <vpux/compiler/conversion/rewriters/generated/convert_IERT2VPUIP.hpp.inc>

//
// ConvertIERT2VPUIPPass
//

class ConvertIERT2VPUIPPass final : public ConvertIERT2VPUIPBase<ConvertIERT2VPUIPPass> {
public:
    explicit ConvertIERT2VPUIPPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    void runOnFunction() final;

public:
    class ConstantRewrite;

private:
    void passBody();

private:
    Logger _log;
};

void ConvertIERT2VPUIPPass::runOnFunction() {
    try {
        _log.trace("Run on Function '@{0}'", getFunction().sym_name());

        passBody();
    } catch (const std::exception& e) {
        printTo(getOperation().emitError(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// ConstantRewrite
//

class ConvertIERT2VPUIPPass::ConstantRewrite final : public mlir::OpRewritePattern<mlir::GetGlobalMemrefOp> {
public:
    ConstantRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::GetGlobalMemrefOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::GetGlobalMemrefOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIERT2VPUIPPass::ConstantRewrite::matchAndRewrite(mlir::GetGlobalMemrefOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Constant Operation '{0}'", origOp);

    auto constantName = origOp.nameAttr();
    if (constantName == nullptr) {
        _log.trace("Failed to get constant name");
        return mlir::failure();
    }

    auto module = origOp.getParentOfType<mlir::ModuleOp>();
    if (module == nullptr) {
        _log.trace("Failed to get parent Module Operation");
        return mlir::failure();
    }

    auto globalOp = module.lookupSymbol<mlir::GlobalMemrefOp>(constantName);
    if (globalOp == nullptr) {
        _log.trace("Failed to get GlobalMemrefOp Operation with name '{0}'", constantName);
        return mlir::failure();
    }

    auto content = globalOp.initial_valueAttr().dyn_cast_or_null<mlir::DenseElementsAttr>();
    if (content == nullptr) {
        _log.trace("GlobalMemrefOp Operation '{0}' has no initial value", constantName);
        return mlir::failure();
    }

    auto memrefType = globalOp.type().dyn_cast_or_null<mlir::MemRefType>();
    VPUX_THROW_UNLESS(memrefType != nullptr, "GlobalMemrefOp Operation '{0}' has unsupported Type '{1}'", constantName,
                      globalOp.type());

    rewriter.replaceOpWithNewOp<VPUIP::DeclareConstantTensorOp>(origOp, memrefType, content);

    _log.trace("Replaced with 'VPUIP.DeclareConstantTensorOp'");

    return mlir::success();
}

//
// passBody
//

void ConvertIERT2VPUIPPass::passBody() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalOp<IERT::CNNNetworkOp, IERT::DataInfoOp, IERT::EndOp>();
    target.addLegalOp<IERT::RunTimeResourcesOp, IERT::MemoryResourceOp, IERT::ExecutorResourceOp>();
    target.addLegalOp<mlir::AllocOp, mlir::DeallocOp, mlir::GlobalMemrefOp>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

    mlir::OwningRewritePatternList patterns;
    patterns.insert<ConstantRewrite>(&ctx, _log.nest());
    populateWithGenerated(&ctx, patterns);

    auto func = getFunction();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLowerIERT2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertIERT2VPUIPPass(Logger log) {
    return std::make_unique<ConvertIERT2VPUIPPass>(log);
}

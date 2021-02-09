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

#include <mlir/Dialect/Quant/QuantTypes.h>
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
    class FakeQuantizeRewrite;

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
        errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

//
// ConstantRewrite
//

class ConvertIERT2VPUIPPass::ConstantRewrite final : public mlir::OpRewritePattern<IERT::ConstantOp> {
public:
    ConstantRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IERT::ConstantOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::ConstantOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIERT2VPUIPPass::ConstantRewrite::matchAndRewrite(IERT::ConstantOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Constant Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPUIP::DeclareConstantTensorOp>(origOp, origOp.getType(), origOp.value());
    _log.trace("Replaced with 'VPUIP.DeclareConstantTensorOp'");

    return mlir::success();
}

//
// FakeQuantizeRewrite
//

class ConvertIERT2VPUIPPass::FakeQuantizeRewrite final : public mlir::OpRewritePattern<IERT::FakeQuantizeOp> {
public:
    FakeQuantizeRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IERT::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIERT2VPUIPPass::FakeQuantizeRewrite::matchAndRewrite(IERT::FakeQuantizeOp origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Found FakeQuantize Operation '{0}'", origOp->getLoc());

    auto inLowConst = origOp.input_low().getDefiningOp<ConstantInterface>();
    auto inHighConst = origOp.input_high().getDefiningOp<ConstantInterface>();
    auto outLowConst = origOp.output_low().getDefiningOp<ConstantInterface>();
    auto outHighConst = origOp.output_high().getDefiningOp<ConstantInterface>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        _log.trace("[{0}] Got non constant parameters \n", origOp->getLoc());
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPUIP::FakeQuantizeUPAOp>(origOp, origOp.input(), origOp.output(), origOp.levels(),
                                                          inLowConst.getContent(), inHighConst.getContent(),
                                                          outLowConst.getContent(), outHighConst.getContent());

    return mlir::success();
}

//
// passBody
//

void ConvertIERT2VPUIPPass::passBody() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalOp<IE::CNNNetworkOp, IE::DataInfoOp, IE::EndOp>();
    target.addLegalOp<IERT::RunTimeResourcesOp, IERT::MemoryResourceOp, IERT::ExecutorResourceOp>();
    target.addLegalOp<mlir::AllocOp, mlir::DeallocOp>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

    mlir::OwningRewritePatternList patterns;
    patterns.insert<ConstantRewrite>(&ctx, _log.nest());
    patterns.insert<FakeQuantizeRewrite>(&ctx, _log.nest());
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

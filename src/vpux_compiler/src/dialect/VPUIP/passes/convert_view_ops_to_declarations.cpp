//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

namespace {

//
// ViewLikeRewrite
//

class ViewLikeRewrite final : public mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface> {
public:
    ViewLikeRewrite(mlir::MLIRContext* ctx, const AliasesInfo* aliasInfo, Logger log)
            : mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface>(ctx), _aliasInfo(aliasInfo), _log(log) {
        VPUX_THROW_UNLESS(_aliasInfo != nullptr, "Got NULL pointer for AliasesInfo in ViewLikeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::ViewLikeOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Byte calculateOffset(mlir::Value val) const;

private:
    const AliasesInfo* _aliasInfo = nullptr;
    Logger _log;
};

Byte ViewLikeRewrite::calculateOffset(mlir::Value val) const {
    Byte offset(0);

    if (auto source = _aliasInfo->getSource(val)) {
        offset = calculateOffset(source);
    }

    if (auto declareOp = mlir::dyn_cast_or_null<VPURT::DeclareBufferOp>(val.getDefiningOp())) {
        offset += Byte(declareOp.byteOffset());
    }

    if (auto subViewOp = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(val.getDefiningOp())) {
        const auto strides = getStrides(subViewOp.source());
        const auto offsets = parseIntArrayAttr<int64_t>(subViewOp.static_offsets());
        VPUX_THROW_UNLESS(strides.size() == offsets.size(), "SubView offsets '{0}' doesn't match strides '{1}'",
                          offsets, strides);

        for (auto p : zip(strides, offsets)) {
            offset += Byte(std::get<0>(p) * std::get<1>(p));
        }
    }

    return offset;
}

mlir::LogicalResult ViewLikeRewrite::matchAndRewrite(mlir::ViewLikeOpInterface origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    if (!mlir::isa<VPUIP::GenericReshapeOp, VPUIP::SubViewOp, VPUIP::PermuteCastOp, VPUIP::QuantizeCastOp,
                   VPUIP::DistributedCastOp, VPUIP::ShapeCastOp>(origOp.getOperation())) {
        return matchFailed(rewriter, origOp, "Unknown view-like operation '{0}'", origOp->getName());
    }

    _log.trace("Found view-like Operation '{0}'", origOp->getLoc());

    const auto origVal = origOp->getResult(0);
    const Byte offset = calculateOffset(origVal);

    const auto roots = _aliasInfo->getRoots(origVal);
    VPUX_THROW_UNLESS(roots.size() == 1, "Value '{0}' expected to have only one root. Got {1}", origVal, roots.size());
    const auto rootVal = *roots.begin();

    VPURT::BufferSection section = VPURT::BufferSection::DDR;
    Optional<mlir::ArrayAttr> sectionIndex;

    if (auto declareOp = rootVal.getDefiningOp<VPURT::DeclareBufferOp>()) {
        _log.nest().trace("It aliases internal buffer produced by '{0}'", declareOp->getLoc());

        section = declareOp.section();
        sectionIndex = declareOp.sectionIndex();
    } else if (auto blockArg = rootVal.dyn_cast<mlir::BlockArgument>()) {
        _log.nest().trace("It aliases Block argument '{0}'", blockArg);

        auto funcOp = mlir::dyn_cast_or_null<mlir::FuncOp>(blockArg.getOwner()->getParentOp());
        VPUX_THROW_UNLESS(funcOp != nullptr, "The view source doesn't belong to Function");

        const auto argInd = checked_cast<size_t>(blockArg.getArgNumber());

        const auto numOutputs = funcOp.getNumResults();
        VPUX_THROW_UNLESS(numOutputs < funcOp.getNumArguments(), "The Function '@{0}' is not bufferized",
                          funcOp.getName());

        size_t numProfilingOutputs = 0;
        if (auto module = blockArg.getParentRegion()->getParentOfType<mlir::ModuleOp>()) {
            auto netOps = to_small_vector(module.getOps<IE::CNNNetworkOp>());
            if (!netOps.empty()) {
                numProfilingOutputs = netOps.front().getProfilingOutputsCount();
            }
        }
        const auto numNetOutputs = numOutputs - numProfilingOutputs;
        const auto numNetInputs = funcOp.getNumArguments() - numOutputs;

        int64_t sectionIndexVal;
        if (argInd < numNetInputs) {
            _log.nest(2).trace("It aliases network input");

            section = VPURT::BufferSection::NetworkInput;
            sectionIndexVal = argInd;
        } else if (argInd < numNetInputs + numNetOutputs) {
            _log.nest(2).trace("It aliases network output");

            section = VPURT::BufferSection::NetworkOutput;
            sectionIndexVal = argInd - numNetInputs;
        } else if (argInd < numNetInputs + numOutputs) {
            _log.nest(2).trace("It aliases network output");

            section = VPURT::BufferSection::ProfilingOutput;
            sectionIndexVal = argInd - numNetInputs - numNetOutputs;
        } else {
            VPUX_THROW("The view source doesn't belong to network entry point Function");
        }
        sectionIndex = getIntArrayAttr(getContext(), makeArrayRef(sectionIndexVal));
    } else {
        VPUX_THROW("Unknown source owner");
    }

    const auto outType = origOp->getResult(0).getType();

    if (sectionIndex.hasValue()) {
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, outType, section, sectionIndex.getValue(),
                                                            offset.count(), nullptr);
    } else {
        rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, outType, section, offset.count());
    }

    return mlir::success();
}

class RewriteConcatView final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    RewriteConcatView(::mlir::MLIRContext* ctx): mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult RewriteConcatView::matchAndRewrite(VPUIP::ConcatViewOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    for (auto input : origOp.inputs()) {
        if (auto waitOp = input.getDefiningOp<mlir::async::AwaitOp>()) {
            if (waitOp->hasOneUse()) {
                waitOp->dropAllUses();
                waitOp->erase();
            }
        }
    }

    rewriter.replaceOp(origOp, origOp.output_buff());
    return ::mlir::success();
};

//
// ConvertViewOpsToDeclarationsPass
//

class ConvertViewOpsToDeclarationsPass final :
        public VPUIP::ConvertViewOpsToDeclarationsBase<ConvertViewOpsToDeclarationsPass> {
public:
    explicit ConvertViewOpsToDeclarationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertViewOpsToDeclarationsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto& aliasInfo = getAnalysis<AliasesInfo>();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<mlir::async::AsyncDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addIllegalOp<VPUIP::GenericReshapeOp, VPUIP::SubViewOp, VPUIP::ConcatViewOp, VPUIP::PermuteCastOp,
                        VPUIP::QuantizeCastOp, VPUIP::DistributedCastOp, VPUIP::ShapeCastOp>();

    target.addLegalOp<VPUIP::SwKernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ViewLikeRewrite>(&ctx, &aliasInfo, _log);
    patterns.insert<RewriteConcatView>(&ctx);

    auto func = getFunction();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertViewOpsToDeclarationsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertViewOpsToDeclarationsPass(Logger log) {
    return std::make_unique<ConvertViewOpsToDeclarationsPass>(log);
}

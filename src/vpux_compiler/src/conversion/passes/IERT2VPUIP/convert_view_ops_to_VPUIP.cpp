//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Transforms/DialectConversion.h>

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

    if (auto declareOp = mlir::dyn_cast_or_null<VPUIP::DeclareTensorOp>(val.getDefiningOp())) {
        offset += Byte(declareOp.dataIndex());
    }

    if (auto subViewOp = mlir::dyn_cast_or_null<IERT::SubViewOp>(val.getDefiningOp())) {
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
    if (!mlir::isa<IERT::GenericReshapeOp, IERT::SubViewOp, IERT::ImplicitReorderOp>(origOp.getOperation())) {
        return matchFailed(rewriter, origOp, "Unknown view-like operation '{0}'", origOp->getName());
    }

    _log.trace("Found view-like Operation '{0}'", origOp->getLoc());

    const auto origVal = origOp->getResult(0);
    const Byte dataOffset = calculateOffset(origVal);

    const auto rootVal = _aliasInfo->getRoot(origVal);

    VPUIP::MemoryLocation locale = VPUIP::MemoryLocation::VPU_DDR_Heap;
    SmallVector<int64_t> localeIndex;

    if (auto declareOp = rootVal.getDefiningOp<VPUIP::DeclareTensorOp>()) {
        _log.nest().trace("It aliases internal buffer produced by '{0}'", declareOp->getLoc());

        locale = declareOp.locale();
        localeIndex = parseIntArrayAttr<int64_t>(declareOp.localeIndex());
    } else if (auto blockArg = rootVal.dyn_cast<mlir::BlockArgument>()) {
        _log.nest().trace("It aliases Block argument '{0}'", blockArg);

        auto funcOp = mlir::dyn_cast_or_null<mlir::FuncOp>(blockArg.getOwner()->getParentOp());
        VPUX_THROW_UNLESS(funcOp != nullptr, "The view source doesn't belong to Function");

        const auto argInd = checked_cast<size_t>(blockArg.getArgNumber());

        const auto numNetOutputs = funcOp.getNumResults();
        VPUX_THROW_UNLESS(numNetOutputs < funcOp.getNumArguments(), "The Function '@{0}' is not bufferized",
                          funcOp.getName());

        const auto numNetInputs = funcOp.getNumArguments() - numNetOutputs;

        if (argInd < numNetInputs) {
            _log.nest(2).trace("It aliases network input");

            locale = VPUIP::MemoryLocation::ProgrammableInput;
            localeIndex.push_back(argInd);
        } else if (argInd < numNetInputs + numNetOutputs) {
            _log.nest(2).trace("It aliases network output");

            locale = VPUIP::MemoryLocation::ProgrammableOutput;
            localeIndex.push_back(argInd - numNetInputs);
        } else {
            VPUX_THROW("The view source doesn't belong to network entry point Function");
        }
    } else {
        VPUX_THROW("Unknown source owner");
    }

    const auto outType = origOp->getResult(0).getType();
    rewriter.replaceOpWithNewOp<VPUIP::DeclareTensorOp>(origOp, outType, locale, localeIndex, dataOffset.count());

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/conversion/rewriters/generated/convert_view_ops_to_VPUIP.hpp.inc>

//
// ConvertViewOps2VPUIPPass
//

class ConvertViewOps2VPUIPPass final : public ConvertViewOps2VPUIPBase<ConvertViewOps2VPUIPPass> {
public:
    explicit ConvertViewOps2VPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertViewOps2VPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto& aliasInfo = getAnalysis<AliasesInfo>();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<mlir::async::AsyncDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();

    target.addLegalOp<VPUIP::SW_KernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SW_KernelOp>([&](mlir::Operation*) {
        return true;
    });


    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ViewLikeRewrite>(&ctx, &aliasInfo, _log);
    populateWithGenerated(patterns);

    auto func = getFunction();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertViewOps2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertViewOps2VPUIPPass(Logger log) {
    return std::make_unique<ConvertViewOps2VPUIPPass>(log);
}

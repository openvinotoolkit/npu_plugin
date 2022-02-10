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

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux {
namespace VPU {

namespace {

mlir::Value copyIntoMemSpace(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val,
                             vpux::IndexedSymbolAttr destinationMemSpace) {
    return builder.createOrFold<IE::CopyOp>(loc, val, destinationMemSpace);
}

//
// CopiesForNCEOp
//

class CopiesForNCEOp final : public mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface> {
public:
    CopiesForNCEOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface>(ctx), _log(log) {
        setDebugName("CopiesForNCEOp");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CopiesForNCEOp::matchAndRewrite(VPU::NCEOpInterface origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto memSpaceCMX = IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(MemoryKind::CMX_NN));

    llvm::DenseMap<mlir::Value, mlir::Value> copiedInputs;
    for (const auto inputOperandIdx : irange(origOp->getNumOperands())) {
        auto& inputOperand = origOp->getOpOperand(inputOperandIdx);
        auto origInputValue = inputOperand.get();
        // No need to copy the data if due to some reason it's in CMX already
        const auto inputMemSpace = IE::getMemorySpace(origInputValue.getType().cast<mlir::RankedTensorType>());
        if (inputMemSpace == memSpaceCMX) {
            continue;
        }
        if (copiedInputs.count(origInputValue) == 0) {
            const auto locationSuffix = llvm::formatv("input-{0}-CMX", inputOperandIdx).str();
            const auto inputCMX = copyIntoMemSpace(rewriter, appendLoc(origOp->getLoc(), locationSuffix),
                                                   origInputValue, memSpaceCMX);
            copiedInputs[origInputValue] = inputCMX;
        }
        inputOperand.set(copiedInputs[origInputValue]);
    }

    auto origOutput = origOp->getResult(0);
    const auto origOutType = origOutput.getType();
    const auto origOutMemSpace = IE::getMemorySpace(origOutType.cast<mlir::RankedTensorType>());
    if (origOutMemSpace != memSpaceCMX) {
        const auto newOutType = changeMemSpace(origOutType.cast<mlir::RankedTensorType>(), VPU::MemoryKind::CMX_NN);
        origOutput.setType(newOutType);
        const auto copiedOutput =
                copyIntoMemSpace(rewriter, appendLoc(origOp->getLoc(), "output-DDR"), origOutput, origOutMemSpace);
        origOutput.replaceAllUsesExcept(copiedOutput, copiedOutput.getDefiningOp());
        rewriter.replaceOp(origOp, copiedOutput);
    }

    return mlir::success();
}

//
// AdjustMemorySpacePass
//

class AdjustMemorySpacePass final : public AdjustMemorySpaceBase<AdjustMemorySpacePass> {
public:
    explicit AdjustMemorySpacePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void AdjustMemorySpacePass::safeRunOnFunc() {
    auto& ctx = getContext();

    // NCE operations are only legal if all their outputs and inputs (incl. weights) reside in CMX
    const auto isLegalOp = [&](mlir::Operation* op) {
        if (auto iface = mlir::dyn_cast<VPU::NCEOpInterface>(op)) {
            const auto memSpaceCMX = IndexedSymbolAttr::get(&ctx, stringifyEnum(MemoryKind::CMX_NN));
            for (auto operand : iface->getOperands()) {
                if (IE::getMemorySpace(operand.getType().cast<mlir::RankedTensorType>()) != memSpaceCMX) {
                    return false;
                }
            }
            for (auto result : iface->getResults()) {
                if (IE::getMemorySpace(result.getType().cast<mlir::RankedTensorType>()) != memSpaceCMX) {
                    return false;
                }
            }
        }
        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal(isLegalOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<CopiesForNCEOp>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustMemorySpacePass
//

std::unique_ptr<mlir::Pass> createAdjustMemorySpacePass(Logger log) {
    return std::make_unique<AdjustMemorySpacePass>(log);
}

}  // namespace VPU
}  // namespace vpux

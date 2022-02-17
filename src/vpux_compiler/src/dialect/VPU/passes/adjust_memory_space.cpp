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
    for (auto& inputOperand : origOp->getOpOperands()) {
        auto origInputValue = inputOperand.get();
        // No need to copy the data if due to some reason it's in CMX already
        const auto inputMemSpace = origInputValue.getType().cast<vpux::NDTypeInterface>().getMemSpace();
        if (inputMemSpace == memSpaceCMX) {
            continue;
        }

        /// Make sure that we copy each piece of data into CMX only once
        /// @example
        /// Bad:
        ///   Input --> Copy -> NCEEltwise(Abs)
        ///        \--> Copy --/
        /// OK:
        ///   Input -> Copy -> NCEEltwise(Abs)
        ///                \--/
        if (copiedInputs.count(origInputValue) == 0) {
            const auto locationSuffix = llvm::formatv("input-{0}-CMX", inputOperand.getOperandNumber()).str();
            const auto inputCMX = copyIntoMemSpace(rewriter, appendLoc(origOp->getLoc(), locationSuffix),
                                                   origInputValue, memSpaceCMX);
            copiedInputs[origInputValue] = inputCMX;
        }
        inputOperand.set(copiedInputs[origInputValue]);
    }

    auto origOutput = origOp->getResult(0);
    const auto origOutType = origOutput.getType().cast<vpux::NDTypeInterface>();
    const auto origOutMemSpace = origOutType.getMemSpace();
    if (origOutMemSpace != memSpaceCMX) {
        // Leave the original operation but change it in-place and add a Copy after it
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(origOp);
        const auto newOutType = origOutType.changeMemSpace(VPU::MemoryKind::CMX_NN);
        rewriter.updateRootInPlace(origOp, [&]() {
            origOutput.setType(newOutType);
            const auto copiedOutput =
                    copyIntoMemSpace(rewriter, appendLoc(origOp->getLoc(), "output-DDR"), origOutput, origOutMemSpace);
            origOutput.replaceAllUsesExcept(copiedOutput, copiedOutput.getDefiningOp());
        });
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
    const auto isLegalOp = [](mlir::Operation* op) {
        if (auto nce_op = mlir::dyn_cast<VPU::NCEOpInterface>(op)) {
            const auto verify_location_in_cmx = [](mlir::Value operand) {
                return operand.getType().cast<vpux::NDTypeInterface>().getMemoryKind() == MemoryKind::CMX_NN;
            };
            return llvm::all_of(nce_op->getOperands(), verify_location_in_cmx) &&
                   llvm::all_of(nce_op->getResults(), verify_location_in_cmx);
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

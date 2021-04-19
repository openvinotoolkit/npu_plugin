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

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// Generated
//

#include <vpux/compiler/conversion/rewriters/generated/lower_IERT_to_VPUIP.hpp.inc>

//
// LowerIERT2VPUIPPass
//

class LowerIERT2VPUIPPass final : public LowerIERT2VPUIPBase<LowerIERT2VPUIPPass> {
public:
    explicit LowerIERT2VPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ConstantRewrite;
    class CTCGreedyDecoderSeqLenRewrite;
    class FakeQuantizeRewrite;
    class ViewLikeRewrite;

private:
    void safeRunOnFunc() final;
};

//
// ConstantRewrite
//

class LowerIERT2VPUIPPass::ConstantRewrite final : public mlir::OpRewritePattern<IERT::ConstantOp> {
public:
    ConstantRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IERT::ConstantOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::ConstantOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LowerIERT2VPUIPPass::ConstantRewrite::matchAndRewrite(IERT::ConstantOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Constant Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPUIP::DeclareConstantTensorOp>(origOp, origOp.getType(), origOp.value());
    _log.trace("Replaced with 'VPUIP.DeclareConstantTensorOp'");

    return mlir::success();
}

//
// CTCGreedyDecoderSeqLenRewrite
//

class LowerIERT2VPUIPPass::CTCGreedyDecoderSeqLenRewrite final :
        public mlir::OpRewritePattern<IERT::CTCGreedyDecoderSeqLenOp> {
public:
    CTCGreedyDecoderSeqLenRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IERT::CTCGreedyDecoderSeqLenOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::CTCGreedyDecoderSeqLenOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LowerIERT2VPUIPPass::CTCGreedyDecoderSeqLenRewrite::matchAndRewrite(
        IERT::CTCGreedyDecoderSeqLenOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found CTCGreedyDecoderSeqLen Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPUIP::CTCGreedyDecoderSeqLenUPAOp>(
            origOp, origOp.input(), origOp.sequenceLength(), origOp.blankIndex(), origOp.output_buff(),
            origOp.outputLength_buff(), origOp.mergeRepeatedAttr());
    _log.trace("Replaced with 'VPUIP.CTCGreedyDecoderSeqLenOp'");

    return mlir::success();
}

//
// FakeQuantizeRewrite
//

class LowerIERT2VPUIPPass::FakeQuantizeRewrite final : public mlir::OpRewritePattern<IERT::FakeQuantizeOp> {
public:
    FakeQuantizeRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IERT::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LowerIERT2VPUIPPass::FakeQuantizeRewrite::matchAndRewrite(IERT::FakeQuantizeOp origOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Found FakeQuantize Operation '{0}'", origOp->getLoc());

    auto inLowConst = origOp.input_low().getDefiningOp<ConstantInterface>();
    auto inHighConst = origOp.input_high().getDefiningOp<ConstantInterface>();
    auto outLowConst = origOp.output_low().getDefiningOp<ConstantInterface>();
    auto outHighConst = origOp.output_high().getDefiningOp<ConstantInterface>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return matchFailed(rewriter, origOp, "Got non constant parameters");
    }

    rewriter.replaceOpWithNewOp<VPUIP::FakeQuantizeUPAOp>(origOp, origOp.input(), origOp.output_buff(), origOp.levels(),
                                                          inLowConst.getContent(), inHighConst.getContent(),
                                                          outLowConst.getContent(), outHighConst.getContent());

    return mlir::success();
}

//
// ViewLikeRewrite
//

class LowerIERT2VPUIPPass::ViewLikeRewrite final : public mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface> {
public:
    ViewLikeRewrite(IE::CNNNetworkOp netInfo, mlir::FuncOp netFunc, AliasesInfo* aliasInfo, Logger log)
            : mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface>(netInfo.getContext()),
              _netInfo(netInfo),
              _netFunc(netFunc),
              _aliasInfo(aliasInfo),
              _log(log) {
        VPUX_THROW_UNLESS(aliasInfo != nullptr, "Got NULL pointer in ViewLikeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::ViewLikeOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mutable IE::CNNNetworkOp _netInfo;
    mutable mlir::FuncOp _netFunc;
    AliasesInfo* _aliasInfo = nullptr;
    Logger _log;
};

mlir::LogicalResult LowerIERT2VPUIPPass::ViewLikeRewrite::matchAndRewrite(mlir::ViewLikeOpInterface origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    if (!mlir::isa<IERT::GenericReshapeOp, mlir::linalg::ReshapeOp, mlir::memref::SubViewOp>(origOp.getOperation())) {
        return matchFailed(rewriter, origOp, "Unknown view-like operation '{0}'", origOp->getName());
    }

    _log.trace("Found view-like Operation '{0}'", origOp->getLoc());

    if (origOp->getParentOfType<mlir::FuncOp>() != _netFunc) {
        return matchFailed(rewriter, origOp, "The operation doesn't belong to network entry point Function");
    }

    const auto origInput = origOp.getViewSource();
    const auto rootVal = _aliasInfo->getRoot(origInput);

    VPUIP::MemoryLocation location = VPUIP::MemoryLocation::VPU_DDR_Heap;
    Optional<uint32_t> locationIndex;
    Byte viewOffset(0);

    if (auto allocOp = rootVal.getDefiningOp<IERT::StaticAllocOp>()) {
        _log.nest().trace("It aliases internal buffer produced by '{0}' StaticAlloc", allocOp.getLoc());

        // TODO: generalize location type
        location = VPUIP::MemoryLocation::VPU_DDR_Heap;
        viewOffset = Byte(allocOp.offset());
    } else if (auto blockArg = rootVal.dyn_cast<mlir::BlockArgument>()) {
        _log.nest().trace("It aliases internal Function argument");

        if (blockArg.getOwner()->getParentOp() != _netFunc) {
            return matchFailed(rewriter, origOp, "The view source doesn't belong to network entry point Function");
        }

        const auto argInd = checked_cast<size_t>(blockArg.getArgNumber());
        const auto numNetInputs = _netInfo.getNetInputsCount();
        const auto numNetOutputs = _netInfo.getNetOutputsCount();

        if (argInd < numNetInputs) {
            _log.nest(2).trace("It aliases network input");

            location = VPUIP::MemoryLocation::ProgrammableInput;
            locationIndex = checked_cast<uint32_t>(argInd);
        } else if (argInd < numNetInputs + numNetOutputs) {
            _log.nest(2).trace("It aliases network output");

            location = VPUIP::MemoryLocation::ProgrammableOutput;
            locationIndex = checked_cast<uint32_t>(argInd - numNetInputs);
        } else {
            return matchFailed(rewriter, origOp, "The view source doesn't belong to network entry point Function");
        }
    } else {
        return matchFailed(rewriter, origOp, "Unknown source owner");
    }

    if (auto subViewOp = mlir::dyn_cast<mlir::memref::SubViewOp>(origOp.getOperation())) {
        int64_t subviewOffset = 0;
        SmallVector<int64_t> resultStrides;
        if (mlir::getStridesAndOffset(subViewOp.getType(), resultStrides, subviewOffset).failed()) {
            return errorAt(origOp, "Can't extract offsets and strides from SubView");
        }

        // Add offset for subview memref in bytes
        const Byte elemSize = getElemTypeSize(subViewOp.getType().getElementType());
        viewOffset += subviewOffset * elemSize;
    }

    const auto outType = origOp->getResult(0).getType();

    if (locationIndex.hasValue()) {
        rewriter.replaceOpWithNewOp<VPUIP::DeclareTensorOp>(origOp, outType, location, locationIndex.getValue(),
                                                            viewOffset.count());
    } else {
        rewriter.replaceOpWithNewOp<VPUIP::DeclareTensorOp>(origOp, outType, location, viewOffset.count());
    }

    return mlir::success();
}

//
// safeRunOnFunc
//

void LowerIERT2VPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    auto& aliasInfo = getAnalysis<AliasesInfo>();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(module != nullptr, "Can't get module from Function '{0}'", func.getLoc());

    IE::CNNNetworkOp netInfo;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netInfo, netFunc);

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<IE::IEDialect>();
    target.addIllegalDialect<IERT::IERTDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalOp<IE::CNNNetworkOp, IE::DataInfoOp>();
    target.addLegalOp<IERT::RunTimeResourcesOp, IERT::MemoryResourceOp, IERT::ExecutorResourceOp>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<mlir::ModuleOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConstantRewrite>(&ctx, _log);
    patterns.insert<CTCGreedyDecoderSeqLenRewrite>(&ctx, _log);
    patterns.insert<FakeQuantizeRewrite>(&ctx, _log);
    patterns.insert<ViewLikeRewrite>(netInfo, netFunc, &aliasInfo, _log);
    populateWithGenerated(patterns);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLowerIERT2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createLowerIERT2VPUIPPass(Logger log) {
    return std::make_unique<LowerIERT2VPUIPPass>(log);
}

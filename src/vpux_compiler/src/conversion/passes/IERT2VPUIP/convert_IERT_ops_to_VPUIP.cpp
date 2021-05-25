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

#include <vpux/compiler/conversion/rewriters/generated/convert_IERT_ops_to_VPUIP.hpp.inc>

//
// ConvertIERTOps2VPUIPPass
//

class ConvertIERTOps2VPUIPPass final : public ConvertIERTOps2VPUIPBase<ConvertIERTOps2VPUIPPass> {
public:
    explicit ConvertIERTOps2VPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class CTCGreedyDecoderSeqLenRewrite;
    class FakeQuantizeRewrite;
    class ViewLikeRewrite;

private:
    void safeRunOnFunc() final;
};

//
// CTCGreedyDecoderSeqLenRewrite
//

class ConvertIERTOps2VPUIPPass::CTCGreedyDecoderSeqLenRewrite final :
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

mlir::LogicalResult ConvertIERTOps2VPUIPPass::CTCGreedyDecoderSeqLenRewrite::matchAndRewrite(
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

class ConvertIERTOps2VPUIPPass::FakeQuantizeRewrite final : public mlir::OpRewritePattern<IERT::FakeQuantizeOp> {
public:
    FakeQuantizeRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IERT::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIERTOps2VPUIPPass::FakeQuantizeRewrite::matchAndRewrite(
        IERT::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
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

class ConvertIERTOps2VPUIPPass::ViewLikeRewrite final :
        public mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface> {
public:
    ViewLikeRewrite(mlir::MLIRContext* ctx, AliasesInfo* aliasInfo, Logger log)
            : mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface>(ctx), _aliasInfo(aliasInfo), _log(log) {
        VPUX_THROW_UNLESS(_aliasInfo != nullptr, "Got NULL pointer for AliasesInfo in ViewLikeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::ViewLikeOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    AliasesInfo* _aliasInfo = nullptr;
    Logger _log;
};

mlir::LogicalResult ConvertIERTOps2VPUIPPass::ViewLikeRewrite::matchAndRewrite(mlir::ViewLikeOpInterface origOp,
                                                                               mlir::PatternRewriter& rewriter) const {
    if (!mlir::isa<IERT::GenericReshapeOp, mlir::linalg::ReshapeOp, mlir::memref::SubViewOp>(origOp.getOperation())) {
        return matchFailed(rewriter, origOp, "Unknown view-like operation '{0}'", origOp->getName());
    }

    _log.trace("Found view-like Operation '{0}'", origOp->getLoc());

    const auto origInput = origOp.getViewSource();
    const auto rootVal = _aliasInfo->getRoot(origInput);

    VPUIP::MemoryLocation location;
    Optional<uint32_t> locationIndex;
    Byte viewOffset(0);

    if (auto allocOp = rootVal.getDefiningOp<IERT::StaticAllocOp>()) {
        _log.nest().trace("It aliases internal buffer produced by '{0}' StaticAlloc", allocOp.getLoc());

        auto memSpace = allocOp.getType().cast<mlir::MemRefType>();
        auto memLocation = VPUIP::getMemoryLocation(memSpace);
        location = (mlir::failed(memLocation)) ? VPUIP::MemoryLocation::VPU_DDR_Heap : memLocation.getValue();

        viewOffset = Byte(allocOp.offset());
    } else if (auto blockArg = rootVal.dyn_cast<mlir::BlockArgument>()) {
        _log.nest().trace("It aliases internal Function argument");

        auto funcOp = mlir::dyn_cast_or_null<mlir::FuncOp>(blockArg.getOwner()->getParentOp());
        if (funcOp == nullptr) {
            return matchFailed(rewriter, origOp, "The view source doesn't belong to Function");
        }

        const auto argInd = checked_cast<size_t>(blockArg.getArgNumber());

        const auto numNetOutputs = funcOp.getNumResults();
        if (numNetOutputs >= funcOp.getNumArguments()) {
            return matchFailed(rewriter, origOp, "The Function '@{0}' is not bufferized", funcOp.getName());
        }
        const auto numNetInputs = funcOp.getNumArguments() - numNetOutputs;

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

void ConvertIERTOps2VPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto& aliasInfo = getAnalysis<AliasesInfo>();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CTCGreedyDecoderSeqLenRewrite>(&ctx, _log);
    patterns.insert<FakeQuantizeRewrite>(&ctx, _log);
    patterns.insert<ViewLikeRewrite>(&ctx, &aliasInfo, _log);
    populateWithGenerated(patterns);

    auto func = getFunction();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertIERTOps2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertIERTOps2VPUIPPass(Logger log) {
    return std::make_unique<ConvertIERTOps2VPUIPPass>(log);
}

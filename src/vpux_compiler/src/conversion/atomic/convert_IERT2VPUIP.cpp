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
    class ViewLikeRewrite;
    class CheckUnsupportedTile;

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
        (void)errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
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
        _log.trace("Got non constant parameters");
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPUIP::FakeQuantizeUPAOp>(origOp, origOp.input(), origOp.output(), origOp.levels(),
                                                          inLowConst.getContent(), inHighConst.getContent(),
                                                          outLowConst.getContent(), outHighConst.getContent());

    return mlir::success();
}

//
// ReshapeRewrite
//

class ConvertIERT2VPUIPPass::ViewLikeRewrite final : public mlir::RewritePattern {
public:
    ViewLikeRewrite(IE::CNNNetworkOp netInfo, mlir::FuncOp netFunc, Logger log)
            : mlir::RewritePattern(1, MatchAnyOpTypeTag{}), _netInfo(netInfo), _netFunc(netFunc), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mutable IE::CNNNetworkOp _netInfo;
    mutable mlir::FuncOp _netFunc;
    Logger _log;
};

mlir::LogicalResult ConvertIERT2VPUIPPass::ViewLikeRewrite::matchAndRewrite(mlir::Operation* origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    auto view = mlir::cast<mlir::ViewLikeOpInterface>(origOp);
    if (view == nullptr) {
        return mlir::failure();
    }
    if (origOp->getNumResults() != 1) {
        return mlir::failure();
    }

    _log.trace("Found Reshape Operation '{0}'", origOp->getLoc());

    if (origOp->getParentOfType<mlir::FuncOp>() != _netFunc) {
        _log.nest().trace("It doesn't belong to network entry point Function");
        return mlir::failure();
    }

    const auto origInput = view.getViewSource();
    const auto outType = origOp->getResult(0).getType();

    VPUIP::MemoryLocation location = VPUIP::MemoryLocation::VPU_DDR_Heap;
    Optional<uint32_t> locationIndex;
    uint64_t baseOffset = 0;

    if (auto allocOp = origInput.getDefiningOp<IERT::StaticAllocOp>()) {
        _log.nest().trace("It aliases internal buffer produced by '{0}' StaticAlloc", allocOp.getLoc());

        // TODO: generalize location type
        location = VPUIP::MemoryLocation::VPU_DDR_Heap;
        baseOffset = allocOp.offset();
    } else if (auto blockArg = origInput.dyn_cast<mlir::BlockArgument>()) {
        _log.nest().trace("It aliases internal Function argument");

        if (blockArg.getOwner()->getParentOp() != _netFunc) {
            _log.nest(2).trace("The Block doesn't belong to network entry point Function");
            return mlir::failure();
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
            _log.nest(2).trace("Wrong block argument index '{0}'", argInd);
            return mlir::failure();
        }
    } else {
        _log.nest().trace("Unknown source owner");
        return mlir::failure();
    }

    if (locationIndex.hasValue()) {
        rewriter.replaceOpWithNewOp<VPUIP::DeclareTensorOp>(origOp, outType, location, locationIndex.getValue(),
                                                            baseOffset);
    } else {
        rewriter.replaceOpWithNewOp<VPUIP::DeclareTensorOp>(origOp, outType, location, baseOffset);
    }

    return mlir::success();
}

//
// CheckUnsupportedTile
//

class ConvertIERT2VPUIPPass::CheckUnsupportedTile final : public mlir::OpRewritePattern<IERT::TileOp> {
public:
    CheckUnsupportedTile(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IERT::TileOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::TileOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIERT2VPUIPPass::CheckUnsupportedTile::matchAndRewrite(IERT::TileOp origOp,
                                                                                 mlir::PatternRewriter&) const {
    _log.trace("Found TileOp Operation '{0}'", origOp->getLoc());
    (void)errorAt(origOp, "Tile operation desn't introduced in VPUIP dialect. All TileOp's should be replaced with "
                          "PerAxisTileOp. Please, make shure that `convert-tile-to-per-axis-tiles` is enabled");
    return mlir::failure();
}

//
// passBody
//

void ConvertIERT2VPUIPPass::passBody() {
    auto& ctx = getContext();
    auto func = getFunction();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(module != nullptr, "Can't get module from Function '{0}'", func.getLoc());

    IE::CNNNetworkOp netInfo;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netInfo, netFunc);

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<IE::IEDialect>();
    target.addIllegalDialect<IERT::IERTDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalOp<IE::CNNNetworkOp, IE::DataInfoOp, IE::EndOp>();
    target.addLegalOp<IERT::RunTimeResourcesOp, IERT::MemoryResourceOp, IERT::ExecutorResourceOp>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.insert<ConstantRewrite>(&ctx, _log.nest());
    patterns.insert<FakeQuantizeRewrite>(&ctx, _log.nest());
    patterns.insert<CheckUnsupportedTile>(&ctx, _log.nest());
    patterns.insert<ViewLikeRewrite>(netInfo, netFunc, _log.nest());
    populateWithGenerated(patterns);

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

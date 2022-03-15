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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/IE/loop.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

namespace {

//
// FuseMemPermuteReshapePass
//

class FuseMemPermuteReshapePass final : public IE::FuseMemPermuteReshapeBase<FuseMemPermuteReshapePass> {
public:
    explicit FuseMemPermuteReshapePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class MemPermuteOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ScaleShiftOpConverter
//

class FuseMemPermuteReshapePass::MemPermuteOpConverter final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    MemPermuteOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseMemPermuteReshapePass::MemPermuteOpConverter::matchAndRewrite(
        IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const {


    // MemPermute -> PerMuteCase -> MemPermute -> AffineReshape -> MemPermute 
    // Search reverse
    auto reshape = origOp.input().getDefiningOp<IE::AffineReshapeOp>();
    if (reshape == nullptr) {
        return mlir::failure();
    }
    auto layoutinfo = vpux::IE::getLayoutInfo(reshape);
    _log.nest().trace("Current layouts: {0}", layoutinfo);
    auto mempermute = reshape.input().getDefiningOp<IE::MemPermuteOp>();
    if (mempermute == nullptr) {
        return mlir::failure();
    }

    auto permutecast = mempermute.input().getDefiningOp<IE::PermuteCastOp>();
    if (permutecast == nullptr) {
        return mlir::failure();
    }

    auto first_mempermute = permutecast.input().getDefiningOp<IE::MemPermuteOp>();
    if (first_mempermute == nullptr) {
        return mlir::failure();
    }

    // MemPermute(nchw) -> PermuteCast(nchw)-> MemPermute(nhwc)
    const auto inOrder = DimsOrder::fromValue(first_mempermute.input());
    const auto inShape = getShape(first_mempermute.input());

    const auto outOrder = DimsOrder::fromValue(origOp.output());
    const auto outShape = getShape(origOp.output());

    std::cout<<llvm::formatv("{0} {1}", first_mempermute->getName(), first_mempermute->getLoc()).str()<<std::endl;

    std::cout<<llvm::formatv("Before inOrder {0} {1} outOrder {2} {3}", inOrder, inShape, outOrder, outShape).str()<<std::endl; 
    std::cout<<llvm::formatv("Before inOrder {0} {1} outOrder {2} {3}", inOrder, inShape, outOrder, outShape).str()<<std::endl;

    const auto outShapeAttr = getIntArrayAttr(getContext(), outShape);

    // replace reshape with new reshape
    auto newReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(reshape, first_mempermute.input(), nullptr, false, outShapeAttr);
    layoutinfo = vpux::IE::getLayoutInfo(newReshape);
    _log.nest().trace("newReshape layouts: {0}", layoutinfo);

    // replace mempermute with new mempermute
    SmallVector<uint32_t> perm(4, 0);
    perm[0] = 0;
    perm[1] = 1;
    perm[2] = 2;
    perm[3] = 3;
    auto memPermAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, origOp->getContext()));
    auto dstOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(origOp.getContext()));
    auto newMemPermute = rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(origOp, origOp.input(), dstOrder, memPermAttr);
    layoutinfo = vpux::IE::getLayoutInfo(newMemPermute);
    _log.nest().trace("newMemPermute layouts: {0}", layoutinfo);

    return mlir::success();

}

void FuseMemPermuteReshapePass::safeRunOnFunc() {
    auto func = getFunction();

    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<MemPermuteOpConverter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

}
}  // namespace

//
// createFuseMemPermuteReshapePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseMemPermuteReshapePass(Logger log) {
    return std::make_unique<FuseMemPermuteReshapePass>(log);
}

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

    const auto outOrder = DimsOrder::fromValue(mempermute.output());
    const auto outShape = getShape(mempermute.output());

	std::cout<<llvm::formatv("{0} {1}", first_mempermute->getName(), first_mempermute->getLoc()).str()<<std::endl;

    std::cout<<llvm::formatv("Before inOrder {0} {1} outOrder {2} {3}", inOrder, inShape, outOrder, outShape).str()<<std::endl; 
 	std::cout<<llvm::formatv("Before inOrder {0} {1} outOrder {2} {3}", inOrder, inShape, outOrder, outShape).str()<<std::endl;


    // inOrder == nchw, outOrder == nhwc
    if (inOrder != DimsOrder::NHWC || outOrder != DimsOrder::NCHW){
        // should check mempermute instead of order 
        return mlir::failure();
    }


    // how to build MemPermute Op 
    // auto inOrder = DimsOrder::fromValue(origOp.input());
    // auto outOrder = DimsOrder::fromValue(origOp.output());
    // auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(inOrder, outOrder, origOp->getContext()));
    // rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(origOp, origOp.input(), origOp.dstOrderAttr(), memPermAttr);

    // SmallVector<unsigned> perm(4, 0);
    // for (size_t dimIdx = 0; dimIdx < perm.size(); dimIdx++) {
    //     perm[dimIdx] = checked_cast<unsigned>(dimIdx);
    // }
    // const auto weightsRank = perm.size();
    // if (weightsRank < 2) {
    //     return mlir::failure();
    // }
    // const auto weightsColIdx = weightsRank - 1;
    // const auto weightsRowIdx = weightsRank - 2;
    // perm[weightsColIdx] = checked_cast<unsigned>(weightsRowIdx);
    // perm[weightsRowIdx] = checked_cast<unsigned>(weightsColIdx);
    // const auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, matmulOp->getContext()));

    // create permutation <d0, d1, d2, d3> -> <d0, d2, d3, d1>
    SmallVector<uint32_t> perm(4, 0);
    perm[0] = 0;
    perm[1] = 2;
    perm[2] = 3;
    perm[3] = 1;
    auto memPermAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, origOp->getContext()));

    auto dstOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(origOp.getContext()));

    // auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(inOrder, DimsOrder::NHWC, origOp->getContext()));

 	std::cout<<llvm::formatv("Passed inOrder {0} {1} outOrder {2} {3} {4}", inOrder, inShape, outOrder, outShape, mempermute.dst_orderAttr()).str()<<std::endl;
 	std::cout<<llvm::formatv("Passed memPermAttr {0}", memPermAttr).str()<<std::endl;
 	std::cout<<llvm::formatv("Passed dstOrder {0}", dstOrder).str()<<std::endl;


    auto newMemPermute = rewriter.create<IE::MemPermuteOp>(first_mempermute->getLoc(), first_mempermute.input(), memPermAttr, memPermAttr);

	std::cout<<llvm::formatv("{0} {1}", newMemPermute->getName(), newMemPermute->getLoc()).str()<<std::endl;

    return mlir::failure();


    // rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(first_mempermute, first_mempermute.input(), mempermute.dst_orderAttr(), memPermAttr);

    std::cout << "Replaced with 'IE::MemPermute'" << std::endl;


    // build one reshape op, 

    // return mlir::failure();

    // const auto outputShape = origOp.getType().getShape();
    // const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    // rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, reshape->getOperand(0), nullptr, false, outputShapeAttr);
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

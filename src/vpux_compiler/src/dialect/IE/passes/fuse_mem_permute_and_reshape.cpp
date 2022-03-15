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
    class ReshapeMemPermuteOpConverter;
    class MemPermuteReshapeConverter;

private:
    void safeRunOnFunc() final;
};

//
// FuseMemPermuteReshapePass
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

    // MemPermute -> PerMuteCase -> MemPermute 
    // Search reverse 
    auto permutecast = origOp.input().getDefiningOp<IE::PermuteCastOp>();
    if (permutecast == nullptr) {
        return mlir::failure();
    }

    auto firstMemPermute = permutecast.input().getDefiningOp<IE::MemPermuteOp>();
    if (firstMemPermute == nullptr) {
        return mlir::failure();
    }

    // MemPermute(nchw) -> PermuteCast(nchw)-> MemPermute(nhwc)
    const auto inOrder = DimsOrder::fromValue(firstMemPermute.input());
    // const auto inShape = getShape(firstMemPermute.input());

    const auto outOrder = DimsOrder::fromValue(origOp.output());
    // const auto outShape = getShape(origOp.output());

	// std::cout<<llvm::formatv("{0} {1}", firstMemPermute->getName(), firstMemPermute->getLoc()).str()<<std::endl;

    // std::cout<<llvm::formatv("Before inOrder {0} {1} outOrder {2} {3}", inOrder, inShape, outOrder, outShape).str()<<std::endl; 
 	// std::cout<<llvm::formatv("Before inOrder {0} {1} outOrder {2} {3}", inOrder, inShape, outOrder, outShape).str()<<std::endl;


    // inOrder == nchw, outOrder == nhwc
    if (inOrder != DimsOrder::NHWC || outOrder != DimsOrder::NCHW){
        // should check mempermute instead of order 
        return mlir::failure();
    }


    //Before inOrder NHWC [1, 64, 48, 8] outOrder NCHW [1, 48, 8, 64]
    // create permutation <d0, d1, d2, d3> -> <d0, d2, d3, d1>
    SmallVector<uint32_t> perm(4, 0);
    perm[0] = 0;
    perm[1] = 1;
    perm[2] = 2;
    perm[3] = 3;
    auto memPermAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, origOp->getContext()));

    auto dstOrder = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(origOp.getContext()));

    // auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(inOrder, DimsOrder::NHWC, origOp->getContext()));

 	// std::cout<<llvm::formatv("Passed inOrder {0} {1} outOrder {2} {3} {4}", inOrder, inShape, outOrder, outShape, origOp.dst_orderAttr()).str()<<std::endl;
 	// std::cout<<llvm::formatv("Passed memPermAttr {0}", memPermAttr).str()<<std::endl;
 	// std::cout<<llvm::formatv("Passed dstOrder {0}", dstOrder).str()<<std::endl;


    rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(origOp, firstMemPermute.input(), dstOrder, memPermAttr);
    return mlir::success();

}


//
// ReshapeMemPermuteOpConverter
//

class FuseMemPermuteReshapePass::ReshapeMemPermuteOpConverter final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    ReshapeMemPermuteOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseMemPermuteReshapePass::ReshapeMemPermuteOpConverter::matchAndRewrite(
        IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const {

    // MemPermute -> Reshape -> MemPermute 
    // Search reverse 
    auto reshape = origOp.input().getDefiningOp<IE::AffineReshapeOp>();
    if (reshape == nullptr) {
        return mlir::failure();
    }

    auto firstMemPermute = reshape.input().getDefiningOp<IE::MemPermuteOp>();
    if (firstMemPermute == nullptr) {
        return mlir::failure();
    }

    // MemPermute(nchw) -> PermuteCast(nchw)-> MemPermute(nhwc)
    const auto inOrder = DimsOrder::fromValue(firstMemPermute.input());
    const auto inShape = getShape(firstMemPermute.input());

    const auto outOrder = DimsOrder::fromValue(origOp.output());
    const auto outShape = getShape(origOp.output());

    const auto outShapeAttr = getIntArrayAttr(rewriter.getContext(), outShape);


    std::cout<<llvm::formatv("ReshapeMemPermuteOpConverter {0} {1}", firstMemPermute->getName(), firstMemPermute->getLoc()).str()<<std::endl;
    std::cout<<llvm::formatv("ReshapeMemPermuteOpConverter inOrder {0} {1} outOrder {2} {3}", inOrder, inShape, outOrder, outShapeAttr).str()<<std::endl; 

    // inOrder == nchw, outOrder == nhwc
    if (inOrder != DimsOrder::NHWC || outOrder != DimsOrder::NHWC){
        // should check mempermute instead of order 
        return mlir::failure();
    }


    // const auto outputShape = reshape.getType().getShape();
    // const auto outShapeAttr = getIntArrayAttr(getContext(), outputShape);

    // const auto inShape = reshape.input().getType().cast<mlir::ShapedType>().getShape();
    // const auto reassociationMap = getReassociationMap(inShape, outputShape);

    // // replace reshape with new reshape
    // auto newReshape = rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(reshape, firstMemPermute.input(), getIntArrayOfArray(getContext(), reassociationMap.getValue()), outShapeAttr);
    // auto layoutinfo = vpux::IE::getLayoutInfo(newReshape);
    // _log.nest().trace("newReshape layouts: {0}", layoutinfo);

    // SmallVector<uint32_t> perm(4, 0);
    // perm[0] = 0;
    // perm[1] = 1;
    // perm[2] = 2;
    // perm[3] = 3;
    // auto memPermAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, origOp->getContext()));
    // auto dstOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(origOp.getContext()));
    // auto newMemPermute = rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(origOp, newReshape.output(), dstOrder, memPermAttr);
    // layoutinfo = vpux::IE::getLayoutInfo(newMemPermute);
    // _log.nest().trace("newMemPermute layouts: {0}", layoutinfo);

    return mlir::success();

}


class FuseMemPermuteReshapePass::MemPermuteReshapeConverter final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    MemPermuteReshapeConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseMemPermuteReshapePass::MemPermuteReshapeConverter::matchAndRewrite(
        IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const {

    std::cout<<llvm::formatv("MemPermuteReshapeConverter {0}", origOp->getName()).str()<<std::endl;

    // MemPermute -> PerMuteCase
    // Search reverse 
    auto reshape = origOp.input().getDefiningOp<IE::AffineReshapeOp>();
    if (reshape == nullptr) {
        return mlir::failure();
    }

    std::cout<<llvm::formatv("MemPermuteReshapeConverter {0} {1}", origOp->getName(), reshape->getName()).str()<<std::endl;


    const auto inOrder = DimsOrder::fromValue(reshape.input());
    const auto inShape = getShape(reshape.input());

    const auto outOrder = DimsOrder::fromValue(reshape.output());
    const auto outShape = getShape(reshape.output());


    const auto memInOrder = DimsOrder::fromValue(origOp.input());
    const auto meminShape = getShape(origOp.input());

    const auto memOutOrder = DimsOrder::fromValue(origOp.output());
    const auto memOutShape = getShape(origOp.output());

    std::cout<<llvm::formatv("MemPermuteReshapeConverter AffineReshape In {0} {1}", inOrder, inShape).str()<<std::endl;
    std::cout<<llvm::formatv("MemPermuteReshapeConverter AffineReshape Out {0} {1}", outOrder, outShape).str()<<std::endl;
    std::cout<<llvm::formatv("MemPermuteReshapeConverter memOutShape In {0} {1}", memInOrder, meminShape).str()<<std::endl;
    std::cout<<llvm::formatv("MemPermuteReshapeConverter memOutShape Out {0} {1}", memOutOrder, memOutShape).str()<<std::endl;

    return mlir::failure();

    SmallVector<uint32_t> perm(4, 0);
    perm[0] = 0;
    perm[1] = 1;
    perm[2] = 2;
    perm[3] = 3;
    auto memPermAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, origOp->getContext()));

    // inOrder == nchw, outOrder == nhwc
    if (inOrder != DimsOrder::NHWC || outOrder != DimsOrder::NHWC){
        // should check mempermute instead of order 
        return mlir::failure();
    }

    if (origOp.mem_permAttr() ==  memPermAttr){
        // if it is the same, no change 
        return mlir::failure();
    }

    auto dstOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(origOp.getContext()));

    auto nop = rewriter.replaceOpWithNewOp<IE::MemPermuteOp>(origOp, origOp.input(), dstOrder, memPermAttr);

    std::cout<<llvm::formatv("MemPermuteReshapeConverter New Op{0} {1}", nop.dst_orderAttr(), nop.mem_permAttr()).str()<<std::endl;

    return mlir::success();

}



void FuseMemPermuteReshapePass::safeRunOnFunc() {
    auto func = getFunction();

    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<MemPermuteOpConverter>(&ctx, _log);
    // patterns.insert<ReshapeMemPermuteOpConverter>(&ctx, _log);
    patterns.insert<MemPermuteReshapeConverter>(&ctx, _log);

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

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

#include "vpux/compiler/dialect/IE/passes.hpp"


using namespace vpux;

namespace {

//
// PrefetchTiling
//

//class PrefetchTiling final : public mlir::OpInterfaceRewritePattern<IE::TilingBuilderOpInterface> {
//public:
//    PrefetchTiling(mlir::MLIRContext* ctx, Logger log)
//            : mlir::OpInterfaceRewritePattern<IE::TilingBuilderOpInterface>(ctx), _log(log) {
//        this->setDebugName("PrefetchTiling");
//    }
//    mlir::LogicalResult matchAndRewrite(IE::TilingBuilderOpInterface origOp,
//                                        mlir::PatternRewriter& rewriter) const final;
//
//private:
//    Logger _log;
//};
//
//mlir::LogicalResult PrefetchTiling::matchAndRewrite(IE::TilingBuilderOpInterface origOp,
//                                                   mlir::PatternRewriter& rewriter) const {
//    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());
//    
////    const auto tiles = origOp.generatePrefetchTiling(_log.nest());
//    
//}

//
// PrefetchTilingPass
//
class PrefetchTilingPass final : public IE::PrefetchTilingBase<PrefetchTilingPass> {
public:
    explicit PrefetchTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//
void PrefetchTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    const auto isUnPrefetchable = [this](mlir::Operation* op) {
        // check the prefetch tiling strategy.
        // only consider conv first
        auto convOp = llvm::dyn_cast<IE::ConvolutionOp>(op);
        if (!convOp)
            return true;
        auto sliceOp = op->getOperand(0).getDefiningOp<IE::SliceOp>();
        for (unsigned index = 1; (!sliceOp) && (index < op->getNumOperands()); index++)
            sliceOp = op->getOperand(index).getDefiningOp<IE::SliceOp>();
        if (!sliceOp)
            return true;

        // get isolated tiling axis
        const auto inputShape = getShape(convOp.input());
        const auto outputShape = getShape(convOp.input());
        Shape tileDim(inputShape.size(), 0);
        
        for (unsigned index = 0; index < inputShape.size(); ++index) {
            if (inputShape[Dim(index)] != outputShape[Dim(index)])
                tileDim[Dim(index)] = 1;
        }
        
        // check prefetchable
        if (auto iface = mlir::dyn_cast<IE::TilingInfoOpInterface>(op)) {
          return !iface.supportPrefetchTiling(tileDim, _log.nest());
        }
        return true;
    };
    target.markUnknownOpDynamicallyLegal(isUnPrefetchable);
}


} // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPrefetchTilingPass(Logger log) {
    std::cout<<"=== createPrefetchTilingPass ===" <<std::endl;
    return std::make_unique<PrefetchTilingPass>(log);
}

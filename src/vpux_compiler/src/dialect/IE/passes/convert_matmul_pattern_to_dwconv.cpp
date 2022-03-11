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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul_utils.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace IE {

// In this pass we convert the pattern in ModelF
// to remove redundant UPA Permute layers
//
// input          weights           input          weights    
//   │               │                │               │
//┌──▼──────┐   ┌────▼────┐        ┌──▼──────┐   ┌────▼────┐
//│Transpose│   │ Reshape │        │Transpose│   │ Reshape │
//└──┬──────┘   └────┬────┘        └──┬──────┘   └────┬────┘
//   │               │                │               │ x channelSize 
//┌──▼──────┐   ┌────▼────┐        ┌──▼──────┐   ┌────▼────┐
//│Reshape  │   │ Reshape │        │Reshape  │   │ concat  │  
//└──┬──────┘   └────┬────┘  ===>  └──┬──────┘   └────┬────┘   
//   │               │                │               │
//┌──▼──────┐   ┌────▼────┐        ┌──▼──────┐        │
//│Reshape  │   │ Reshape │        │Reshape  │        │
//└──┬──────┘   └────┬────┘        └──┬──────┘        │
//   │               │                │               │
//┌──▼───────────────▼───┐         ┌──▼───────────────▼───┐
//│        MatMul        │         │        DWConv        │
//└──────────┬───────────┘         └──────────┬───────────┘
//           │                                │
//      ┌────▼────┐                      ┌────▼────┐    
//      │ Reshape │                      │ Reshape │  
//      └────┬────┘                      └────┬────┘     
//           │                                │       
//           │                                │        
//           ▼                                ▼                
// 
//
// MatMulPatternConverter
//

class MatMulPatternConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    using mlir::OpRewritePattern<IE::MatMulOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult MatMulPatternConverter::matchAndRewrite(IE::MatMulOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    if (!checkPermuteMatMulPattern(origOp)) {
        return mlir::failure();
    }

    auto parentOps = getMatMulParentOps(origOp);
    mlir::Operation* origTransposeOp = parentOps[0];
    mlir::Operation* weightInputOp = parentOps[1];

    if (origTransposeOp == nullptr || weightInputOp == nullptr) {
        return mlir::failure();
    }
    VPUX_THROW_UNLESS(mlir::isa<IE::TransposeOp>(origTransposeOp), "MatMul pattern does not match",
                      origTransposeOp->getLoc());

    auto origTransposeShape = origTransposeOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();
    const auto channelSize = origTransposeShape[Dims4D::Act::H] * origTransposeShape[Dims4D::Act::W];
    const auto totalKernelSize = origTransposeShape[Dims4D::Act::C];
    auto kernelFactors = getKernelFactors(totalKernelSize);

    // build the activation
    auto reshape1 = *(origTransposeOp->getResult(0).getUsers().begin());
    VPUX_THROW_WHEN(reshape1 == nullptr, "MatMul pattern mismatch.");
    auto reshape2 = *(reshape1->getResult(0).getUsers().begin());
    VPUX_THROW_WHEN(reshape2 == nullptr, "MatMul pattern mismatch.");
    // replace reshape2
    const auto outShape4DAttr =
            getIntArrayAttr(rewriter.getContext(), Shape({1, kernelFactors[0], kernelFactors[1], channelSize}));
    auto factorReshape =
            rewriter.create<IE::ReshapeOp>(reshape2->getLoc(), reshape1->getResult(0), nullptr, false, outShape4DAttr);
    // insert transpose
    const auto orderAttr =
            mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap({0, 3, 1, 2}, rewriter.getContext()));
    auto transpose =
            rewriter.create<IE::TransposeOp>(factorReshape->getLoc(), factorReshape->getResult(0), nullptr, orderAttr);
    reshape2->replaceAllUsesWith(transpose);
    reshape2->erase();

    // build the weight
    auto weightReshape1 = *(weightInputOp->getResult(0).getUsers().begin());
    VPUX_THROW_WHEN(weightReshape1 == nullptr, "MatMul pattern mismatch.");
    auto weightReshape2 = *(weightReshape1->getResult(0).getUsers().begin());
    VPUX_THROW_WHEN(weightReshape2 == nullptr, "MatMul pattern mismatch.");
    weightReshape1->replaceAllUsesWith(weightReshape2);
    weightReshape1->erase();
    const auto weightOutShape4DAttr =
            getIntArrayAttr(rewriter.getContext(), Shape({1, 1, kernelFactors[0], kernelFactors[1]}));
    auto factorWeightReshape = rewriter.create<IE::ReshapeOp>(weightInputOp->getLoc(), weightInputOp->getResult(0),
                                                              nullptr, false, weightOutShape4DAttr);
    //  concat
    SmallVector<mlir::Value> concatSlices;
    SmallVector<Shape> concatOffsets;
    concatSlices.reserve(channelSize);
    concatOffsets.reserve(channelSize);
    for (auto index = 0; index < channelSize; index++) {
        concatSlices.push_back(factorWeightReshape);
        concatOffsets.push_back(Shape({index, 0, 0, 0, 0}));
    }
    auto concatOp = rewriter.create<IE::ConcatOp>(factorWeightReshape->getLoc(), mlir::ValueRange(concatSlices), 0);
    weightReshape2->replaceAllUsesWith(concatOp);
    weightReshape2->erase();

    // replace matmul
    const SmallVector<int32_t> strides = {1, 1};
    const SmallVector<int32_t> padBegin = {0, 0};
    const SmallVector<int32_t> padEnd = {0, 0};
    const SmallVector<int32_t> dilations = {1, 1};

    auto dilationsAttr = getIntArrayAttr(origOp.getContext(), dilations);
    auto stridesAttr = getIntArrayAttr(origOp.getContext(), strides);
    auto padBeginAttr = getIntArrayAttr(origOp.getContext(), padBegin);
    auto padEndAttr = getIntArrayAttr(origOp.getContext(), padEnd);

    auto groupAttr = getIntAttr(origOp.getContext(), channelSize);

    auto dwconv = rewriter.create<IE::GroupConvolutionOp>(origOp->getLoc(), transpose.output(), concatOp.output(),
                                                          /*bias=*/nullptr, stridesAttr, padBeginAttr, padEndAttr,
                                                          dilationsAttr, groupAttr, nullptr);
    origOp->replaceAllUsesWith(dwconv);
    origOp->erase();
    //
    //    // test
    //    auto tmp = *(reshape1->getResult(0).getUsers().begin());
    //    std::cout << llvm::formatv("\n\nACT:\nafter reshape 1: {0}, {1}, {2}", tmp->getName(), tmp->getLoc(),
    //                               tmp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape())
    //                         .str()
    //              << std::endl;
    //    tmp = *(tmp->getResult(0).getUsers().begin());
    //    std::cout << llvm::formatv("after : {0}, {1}, {2}", tmp->getName(), tmp->getLoc(),
    //                               tmp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape())
    //                         .str()
    //              << std::endl;
    //    tmp = *(tmp->getResult(0).getUsers().begin());
    //    std::cout << llvm::formatv("after : {0}, {1}, {2}\n\n", tmp->getName(), tmp->getLoc(),
    //                               tmp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape())
    //                         .str()
    //              << std::endl;
    //
    //    std::cout << llvm::formatv("\n\nFINAL: \nweightInputOp: {0}, {1}, {2}", weightInputOp->getName(),
    //                               weightInputOp->getLoc(),
    //                               weightInputOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape())
    //                         .str()
    //              << std::endl;
    //    tmp = *(weightInputOp->getResult(0).getUsers().begin());
    //    std::cout << llvm::formatv("after weightInputOp: {0}, {1}, {2}", tmp->getName(), tmp->getLoc(),
    //                               tmp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape())
    //                         .str()
    //              << std::endl;
    //    tmp = *(tmp->getResult(0).getUsers().begin());
    //    std::cout << llvm::formatv("after : {0}, {1}, {2}", tmp->getName(), tmp->getLoc(),
    //                               tmp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape())
    //                         .str()
    //              << std::endl;
    //
    //    tmp = *(tmp->getResult(0).getUsers().begin());
    //    std::cout << llvm::formatv("after : {0}, {1}, {2}", tmp->getName(), tmp->getLoc(),
    //                               tmp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape())
    //                         .str()
    //              << std::endl;
    //    tmp = *(tmp->getResult(0).getUsers().begin());
    //    std::cout << llvm::formatv("after : {0}, {1}, {2}", tmp->getName(), tmp->getLoc(),
    //                               tmp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape())
    //                         .str()
    //              << std::endl;

    return mlir::success();
}

//
// ConvertMatMulPatternToDWConvPass
//

class ConvertMatMulPatternToDWConvPass final :
        public IE::ConvertMatMulPatternToDWConvBase<ConvertMatMulPatternToDWConvPass> {
public:
    explicit ConvertMatMulPatternToDWConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertMatMulPatternToDWConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MatMulPatternConverter>(&ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getFunction(), std::move(patterns),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace IE
}  // namespace vpux

//
// createConvertMatMulPatternToDWConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertMatMulPatternToDWConvPass(Logger log) {
    return std::make_unique<ConvertMatMulPatternToDWConvPass>(log);
}

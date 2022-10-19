//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/passes.hpp"

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

using namespace vpux;

namespace {

//
// OptimizeConcatSliceToSliceConcatPass
//

class OptimizeConcatSliceToSliceConcatPass final :
        public VPU::OptimizeConcatSliceToSliceConcatBase<OptimizeConcatSliceToSliceConcatPass> {
public:
    explicit OptimizeConcatSliceToSliceConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ConcatSliceToSliceConcatConverter;

private:
    void safeRunOnFunc() final;
};

class OptimizeConcatSliceToSliceConcatPass::ConcatSliceToSliceConcatConverter final :
        public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    ConcatSliceToSliceConcatConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::ConcatOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult OptimizeConcatSliceToSliceConcatPass::ConcatSliceToSliceConcatConverter::matchAndRewrite(
        VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto outShape = getShape(origOp.output());

    // Only support 4D Input shape
    if (outShape.size() != 4) {
        return mlir::failure();
    }

    // Do not optimize if there are more than 1 users.
    if (!origOp.output().hasOneUse()) {
        return mlir::failure();
    }

    auto concatConsumer = *origOp.output().user_begin();
    if (!::mlir::isa<VPU::SliceOp>(concatConsumer)) {
        // Do not optimize if the next layer is not SliceOp
        return mlir::failure();
    }
    auto nextSliceOp = mlir::dyn_cast<VPU::SliceOp>(concatConsumer);
    Shape sliceSize = getShape(nextSliceOp.result()).raw();

    if (sliceSize[Dims4D::Act::C] != 1) {
        // TODO: For channel aligned to 1008 and slice channel 1001, the performance may drop.
        // After E47829 is resolved this condition could be removed
        return mlir::failure();
    }

    // Do optimize if all the input layers are VPU::NCEOpInterface ops with tilingStrategy Attribute.
    const auto concatInputList = origOp.inputs();

    const auto hasAllInputNCEopWithTileStrategy =
            !concatInputList.empty() && llvm::all_of(concatInputList, [&](auto input) {
                auto inputOp = input.getDefiningOp();
                return mlir::isa_and_nonnull<VPU::NCEOpInterface>(inputOp) && inputOp->hasAttr(tilingStrategy);
            });

    if (!hasAllInputNCEopWithTileStrategy) {
        _log.trace("Not all inputs are NCE operation with tilingStrategy.");
        return mlir::failure();
    }
    _log.trace("The concat-slice structure could be optimized to slice-concat.");

    Shape concatSize = outShape.raw();

    SmallVector<mlir::Value> newSlices;
    SmallVector<Shape> resultTileOffsets;
    newSlices.reserve(concatInputList.size());
    resultTileOffsets.reserve(concatInputList.size());

    for (auto concatIn : concatInputList) {
        auto concatInShape = getShape(concatIn).raw();
        auto updatedConcatInShape = Shape(concatInShape);
        updatedConcatInShape[Dims4D::Act::C] = sliceSize[Dims4D::Act::C];
        Shape updatedConcatInOffsets = Shape(concatInShape.size(), 0);

        auto subSlice = rewriter.create<VPU::SliceOp>(concatIn.getLoc(), concatIn,
                                                      getIntArrayAttr(concatIn.getContext(), updatedConcatInOffsets),
                                                      getIntArrayAttr(concatIn.getContext(), updatedConcatInShape));

        newSlices.push_back(subSlice);
        resultTileOffsets.push_back(updatedConcatInOffsets);
    }

    rewriter.replaceOpWithNewOp<VPU::ConcatOp>(origOp, nextSliceOp.result().getType(), mlir::ValueRange(newSlices),
                                               origOp.static_offsetsAttr());

    return mlir::success();
}

//
// safeRunOnFunc
//

void OptimizeConcatSliceToSliceConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.add<ConcatSliceToSliceConcatConverter>(&ctx, _log);
    auto func = getFunction();

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeConcatSliceToSliceConcatPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createOptimizeConcatSliceToSliceConcatPass(Logger log) {
    return std::make_unique<OptimizeConcatSliceToSliceConcatPass>(log);
}

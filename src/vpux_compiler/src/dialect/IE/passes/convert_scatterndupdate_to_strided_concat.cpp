//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

class ConvertScatterNDUpdateToStridedConcatPass final :
        public IE::ConvertScatterNDUpdateToStridedConcatBase<ConvertScatterNDUpdateToStridedConcatPass> {
public:
    explicit ConvertScatterNDUpdateToStridedConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ScatterNDUpdateOpConverter;

private:
    void safeRunOnFunc() final;
};

class ConvertScatterNDUpdateToStridedConcatPass::ScatterNDUpdateOpConverter final :
        public mlir::OpRewritePattern<IE::ScatterNDUpdateOp> {
public:
    ScatterNDUpdateOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterNDUpdateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertScatterNDUpdateToStridedConcatPass::ScatterNDUpdateOpConverter::matchAndRewrite(
        IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Get ScatterNDUpdateOp Op {0}", origOp);
    const auto greaterThanOne = [](auto dim) {
        return dim > 1;
    };

    const auto inputShape = getShape(origOp.input());
    const auto indices = origOp.indices();
    const auto indicesShape = getShape(indices);
    auto indicesConst = indices.getDefiningOp<Const::DeclareOp>();
    if (indicesConst == nullptr) {
        return mlir::failure();
    }

    const auto origInType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const int64_t origInRank = origInType.getRank();

    // only optimize elementwise case.
    if (indicesShape[Dim(indicesShape.size() - 1)] != origInRank) {
        return mlir::failure();
    }

    const auto indicesConstValue = indicesConst.content();
    const auto indicesData = indicesConstValue.getValues<int64_t>();

    SmallVector<int64_t> potentialStrides;
    for (int64_t i = 0; i < static_cast<int64_t>(inputShape.size()); i++) {
        // check potential stride.
        // if not integer stride return
        if (inputShape[Dim(i)] % indicesShape[Dim(i)] != 0) {
            return mlir::failure();
        }

        auto strideCandidate = inputShape[Dim(i)] / indicesShape[Dim(i)];
        potentialStrides.push_back(strideCandidate);
    }

    // not 1 dim stride
    if (llvm::count_if(potentialStrides, greaterThanOne) != 1) {
        return mlir::failure();
    }

    auto axis = llvm::find_if(potentialStrides, greaterThanOne);
    VPUX_THROW_UNLESS(axis != potentialStrides.end(), "Can not get correct Axis");

    auto axisIndex = std::distance(potentialStrides.begin(), axis);
    auto stride = potentialStrides[axisIndex];
    auto offsetValue = indicesData[axisIndex];

    // check elementwise indices equal to stride operation.
    // e.g. input shape 1x3x40x40x15, indices 1x3x40x40x5x5, output shape 1x3x40x40x5
    // check indices last dim 5 values could meet offset and stride operation.

    SmallVector<int64_t> strideShape(inputShape.size(), 0);
    strideShape[inputShape.size() - 1] = 1;
    for (const auto ind : irange(inputShape.size() - 1) | reversed) {
        const auto prevDim = ind + 1;
        strideShape[ind] = strideShape[prevDim] * indicesShape[Dim(prevDim)];
    }

    for (int64_t index = 0; index < static_cast<int64_t>(indicesData.size()); index += origInRank) {
        int64_t calculateIndex = 0;
        for (int64_t indiceIndex = 0; indiceIndex < origInRank; indiceIndex++) {
            calculateIndex =
                    (indiceIndex == axisIndex)
                            ? strideShape[indiceIndex] * (indicesData[index + indiceIndex] - offsetValue) / stride +
                                      calculateIndex
                            : strideShape[indiceIndex] * indicesData[index + indiceIndex] + calculateIndex;
        }
        if (calculateIndex != index / origInRank) {
            return mlir::failure();
        }
    }

    auto ctx = origOp.getContext();
    auto zeros = SmallVector<int64_t>(inputShape.size(), 0);
    SmallVector<mlir::Value> subSlices;

    for (const auto ind : irange(stride)) {
        if (ind == offsetValue) {
            subSlices.push_back(origOp.updates());
        } else {
            auto offsetValues = SmallVector<int64_t>(inputShape.size(), 0);
            offsetValues[axisIndex] = ind;

            const auto stridesAttr = getIntArrayAttr(ctx, makeArrayRef(potentialStrides));
            const auto beginsAttr = getIntArrayAttr(ctx, makeArrayRef(offsetValues));
            const auto endsAttr = getIntArrayAttr(ctx, inputShape);
            const auto zeroMask = getIntArrayAttr(ctx, makeArrayRef(zeros));

            auto stridedSliceOp = rewriter.create<IE::StridedSliceOp>(
                    origOp->getLoc(), origOp.input(), nullptr, nullptr, nullptr, beginsAttr, endsAttr, stridesAttr,
                    /*beginMask =*/zeroMask, /*endMask =*/zeroMask, /*newAxisMask =*/zeroMask,
                    /*shrinkAxisMask =*/zeroMask, /*ellipsisMask = */ zeroMask);

            subSlices.push_back(stridedSliceOp);
        }
    }
    auto concatOutput = rewriter.create<IE::ConcatOp>(origOp->getLoc(), subSlices, axisIndex, 1, stride).output();
    rewriter.replaceOp(origOp, concatOutput);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertScatterNDUpdateToStridedConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ScatterNDUpdateOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertScatterNDUpdateToStridedConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertScatterNDUpdateToStridedConcatPass(Logger log) {
    return std::make_unique<ConvertScatterNDUpdateToStridedConcatPass>(log);
}

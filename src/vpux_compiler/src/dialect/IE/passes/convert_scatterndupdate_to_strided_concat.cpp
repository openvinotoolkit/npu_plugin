//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

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
    class ConvertToStridedConcat;
    class ConvertToSliceConcat;

private:
    void safeRunOnFunc() final;
};

class ConvertScatterNDUpdateToStridedConcatPass::ConvertToStridedConcat final :
        public mlir::OpRewritePattern<IE::ScatterNDUpdateOp> {
public:
    ConvertToStridedConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterNDUpdateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// For example, if there is a 1x15 tensor: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
// The Indices to update data is [0,3,6,9,12] . The data to update is [fx0,fx1,fx2,fx3,fx4]
// The results is [fx0,2,3,fx1,5,6,fx2,8,9,fx3,11,12,fx4,14,15].
// It equals to offset 0, stride 3, strided concat.
mlir::LogicalResult ConvertScatterNDUpdateToStridedConcatPass::ConvertToStridedConcat::matchAndRewrite(
        IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Get ScatterNDUpdateOp Op {0}", origOp);
    const auto greaterThanOne = [](auto dim) {
        return dim > 1;
    };

    const auto inputShape = getShape(origOp.getInput());
    const auto indices = origOp.getIndices();
    const auto indicesShape = getShape(indices);
    auto indicesConst = indices.getDefiningOp<Const::DeclareOp>();
    if (indicesConst == nullptr) {
        return mlir::failure();
    }

    const auto origInType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const int64_t origInRank = origInType.getRank();

    // only optimize elementwise case.
    if (indicesShape[Dim(indicesShape.size() - 1)] != origInRank) {
        return mlir::failure();
    }

    const auto indicesConstValue = indicesConst.getContent();
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
    // For example, Input shape 1x10x1, indices shape 1x1x1x3
    // indices data [[[[0, 0, 0]]]]
    // This case will be handled by ConvertToSliceConcat Rewriter with 1 Slice
    // Otherwise it will need 10 Slice by ConvertToStridedConcat Rewriter
    if (indicesShape[Dim(axisIndex)] == 1) {
        return mlir::failure();
    }

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
            subSlices.push_back(origOp.getUpdates());
        } else {
            auto offsetValues = SmallVector<int64_t>(inputShape.size(), 0);
            offsetValues[axisIndex] = ind;

            const auto stridesAttr = getIntArrayAttr(ctx, ArrayRef(potentialStrides));
            const auto beginsAttr = getIntArrayAttr(ctx, ArrayRef(offsetValues));
            const auto endsAttr = getIntArrayAttr(ctx, inputShape);
            const auto zeroMask = getIntArrayAttr(ctx, ArrayRef(zeros));

            auto stridedSliceOp = rewriter.create<IE::StridedSliceOp>(
                    origOp->getLoc(), origOp.getInput(), nullptr, nullptr, nullptr, beginsAttr, endsAttr, stridesAttr,
                    /*beginMask =*/zeroMask, /*endMask =*/zeroMask, /*newAxisMask =*/zeroMask,
                    /*shrinkAxisMask =*/zeroMask, /*ellipsisMask = */ zeroMask);

            subSlices.push_back(stridedSliceOp);
        }
    }
    auto concatOutput = rewriter.create<IE::ConcatOp>(origOp->getLoc(), subSlices, axisIndex, 1, stride).getOutput();
    rewriter.replaceOp(origOp, concatOutput);

    return mlir::success();
}

// ConvertToSliceConcat

class ConvertScatterNDUpdateToStridedConcatPass::ConvertToSliceConcat final :
        public mlir::OpRewritePattern<IE::ScatterNDUpdateOp> {
public:
    ConvertToSliceConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ScatterNDUpdateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const final;

    std::optional<Dim> getUpdateDim(ShapeRef inputShape, ShapeRef indicesShape, Const::DeclareOp indicesConst) const;

private:
    Logger _log;
};

std::optional<Dim> ConvertScatterNDUpdateToStridedConcatPass::ConvertToSliceConcat::getUpdateDim(
        ShapeRef inputShape, ShapeRef indicesShape, Const::DeclareOp indicesConst) const {
    const auto indicesConstValue = indicesConst.getContent();
    const auto indicesData = indicesConstValue.getValues<int64_t>();

    const auto greaterThanOne = [](auto dimSize) {
        return dimSize > 1;
    };

    // Scenario 1: Elements Update
    // For example, Input shape 1x32x1, indices shape 1x3x1x3
    // indices data [[[[0, 5, 0], [0, 6, 0], [0, 7, 0]]]]
    // The updateDim will be Dim(1)
    const auto inputRank = checked_cast<int64_t>(inputShape.size());
    const auto indicesRank = checked_cast<int64_t>(indicesShape.size());
    if (indicesShape.back() == inputRank && indicesRank - 1 == inputRank) {
        const auto inputShapeGreaterThanOne = llvm::count_if(inputShape, greaterThanOne);
        const auto indicesShapeGreaterThanOne =
                std::count_if(indicesShape.begin(), indicesShape.end() - 1, greaterThanOne);
        if (inputShapeGreaterThanOne > 1 || indicesShapeGreaterThanOne > 1) {
            _log.trace("Elements Update: Only support ScatterNDUpdate Op update at one axis");
            return std::nullopt;
        }

        // Input shape 1x1x1, indices shape 1x1x1x3
        // indices data [[[[0, 0, 0]]]]
        // The updateDim will be Dim(0)
        if (inputShapeGreaterThanOne == 0 && indicesShapeGreaterThanOne == 0) {
            return Dim(0);
        }

        auto axis = llvm::find_if(inputShape, greaterThanOne);
        VPUX_THROW_UNLESS(axis != inputShape.end(), "Can not get correct Axis");
        auto updateDim = std::distance(inputShape.begin(), axis);

        const auto beginOffset = indicesData[updateDim];
        for (auto idx = 1; idx < indicesShape[Dim(updateDim)]; idx++) {
            if (indicesData[updateDim + inputRank * idx] != beginOffset + idx) {
                _log.trace("Elements Update: The data in indices and at the updateDim should be increase with step 1");
                return std::nullopt;
            }
        }

        return Dim(updateDim);
    }

    // Scenario 2: Tensor Update
    // For example, Input shape 16x32x64, indices shape 3x1
    // indices data [[5], [6], [7]]
    // The updateDim will be Dim(0)
    if (indicesShape.back() == 1) {
        const auto beginOffset = indicesData.front();
        for (auto idx = 1; idx < indicesShape.totalSize(); idx++) {
            if (indicesData[idx] != beginOffset + idx) {
                _log.trace("Tensor Update: The data in indices and at the updateDim should be increase with step 1");
                return std::nullopt;
            }
        }
        return Dim(0);
    }

    return std::nullopt;
}

// There are two possible patterns can be converted
// Scenario 1: Elements Update, it has the following limitations:
// - indices.shape[-1] = input.shape.rank
// - Only has one updateDim
// - All dim size for indices shape[:-1] should be 1 except the updateDim
// - All dim size for input shape should be 1 except the updateDim
// - The data in indices and at the updateDim should be increase with step 1
// For example, Input shape 1x32x1, indices shape 1x3x1x3
// indices data [[[[0, 5, 0], [0, 6, 0], [0, 7, 0]]]]

// Scenario 2: Tensor Update, it has the following limitations:
// - indices.shape[-1] = 1, if not the update data shape rank will not same with input
// - The data in indices should be increase with step 1
// For example, Input shape 16x32x64, indices shape 3x1
// indices data [[5], [6], [7]]
mlir::LogicalResult ConvertScatterNDUpdateToStridedConcatPass::ConvertToSliceConcat::matchAndRewrite(
        IE::ScatterNDUpdateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto inputShape = getShape(origOp.getInput());
    const auto indices = origOp.getIndices();
    const auto indicesShape = getShape(indices);
    auto indicesConst = indices.getDefiningOp<Const::DeclareOp>();
    if (indicesConst == nullptr) {
        _log.trace("ScatterNDUpdate Op should with constant indices");
        return mlir::failure();
    }

    auto dimValue = getUpdateDim(inputShape, indicesShape, indicesConst);
    if (!dimValue.has_value()) {
        _log.trace("ScatterNDUpdate Op can not convert to Slice and Concat");
        return mlir::failure();
    }
    auto updateDim = dimValue.value().ind();

    const auto indicesConstValue = indicesConst.getContent();
    const auto indicesData = indicesConstValue.getValues<int64_t>();
    auto beginOffset = indicesData[updateDim];

    SmallVector<mlir::Value> concatInputs;
    // Create the left Slice Op
    auto leftSliceOffset = SmallVector<int64_t>(inputShape.size(), 0);
    auto leftSliceShape = to_small_vector(inputShape.raw());
    leftSliceShape[updateDim] = beginOffset;

    if (beginOffset != 0) {
        concatInputs.push_back(
                rewriter.create<IE::SliceOp>(origOp->getLoc(), origOp.getInput(), leftSliceOffset, leftSliceShape)
                        .getResult());
    }

    // Update data value
    concatInputs.push_back(origOp.getUpdates());

    // Create the right Slice Op
    auto endOffset = beginOffset + indicesShape[Dim(updateDim)];
    auto rightSliceOffset = SmallVector<int64_t>(inputShape.size(), 0);
    rightSliceOffset[updateDim] = endOffset;
    auto rightSliceShape = to_small_vector(inputShape.raw());
    rightSliceShape[updateDim] = rightSliceShape[updateDim] - endOffset;

    if (rightSliceShape[updateDim] != 0) {
        concatInputs.push_back(
                rewriter.create<IE::SliceOp>(origOp->getLoc(), origOp.getInput(), rightSliceOffset, rightSliceShape)
                        .getResult());
    }

    _log.trace("Replace '{0}' at '{1}' with Slice and Concat Op", origOp->getName(), origOp->getLoc());
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, concatInputs, updateDim);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertScatterNDUpdateToStridedConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertToStridedConcat>(&ctx, _log);
    patterns.add<ConvertToSliceConcat>(&ctx, _log);

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

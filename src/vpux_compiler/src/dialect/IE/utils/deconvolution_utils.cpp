//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/deconvolution_utils.hpp"

namespace vpux {
namespace IE {
// Checks whether the Deconvolution filter is a constant or a FakeQuantize with a constant input
mlir::FailureOr<Const::DeclareOp> getConstFilter(IE::DeconvolutionOp deconv) {
    if (auto filterFq = deconv.filter().getDefiningOp<IE::FakeQuantizeOp>()) {
        if (auto filterConst = filterFq.input().getDefiningOp<Const::DeclareOp>()) {
            return filterConst;
        }
    } else if (auto filterConst = deconv.filter().getDefiningOp<Const::DeclareOp>()) {
        return filterConst;
    }
    return mlir::failure();
}

mlir::LogicalResult canConvertDeconvToConv(IE::DeconvolutionOp deconv) {
    if (getShape(deconv.feature()).size() != 4) {
        return mlir::failure();
    }

    if (mlir::failed(IE::getConstFilter(deconv))) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult canConvertGroupDeconvToGroupConv(IE::GroupDeconvolutionOp groupDeconv) {
    if (getShape(groupDeconv.feature()).size() != 4) {
        return mlir::failure();
    }

    // Const::DeclareOp - IE::FakeQuantizeOp filter is not handled
    if (!mlir::isa<Const::DeclareOp>(groupDeconv.filter().getDefiningOp())) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::Value createFQ(mlir::PatternRewriter& rewriter, mlir::Value input, IE::FakeQuantizeOp fq) {
    const auto outputType = fq.output().getType().cast<vpux::NDTypeInterface>();
    const auto newOutputType = outputType.changeShape(getShape(input));
    return rewriter
            .create<IE::FakeQuantizeOp>(fq.getLoc(), newOutputType, input, fq.input_low(), fq.input_high(),
                                        fq.output_low(), fq.output_high(), fq.levels(), fq.auto_broadcast())
            ->getResult(0);
}

mlir::Value createPadding(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input, Dim axis,
                          int64_t nums, IE::FakeQuantizeOp inputFQ) {
    auto ctx = rewriter.getContext();

    auto inputShape = getShape(input);
    auto offsets = SmallVector<int64_t>(inputShape.size(), 0);
    auto sizes = SmallVector<int64_t>(inputShape.begin(), inputShape.end());
    offsets[axis.ind()] = inputShape[axis] - 1;
    sizes[axis.ind()] = 1;

    auto subSlice = rewriter.create<IE::SliceOp>(loc, input, getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes))
                            .result();
    if (inputFQ != nullptr) {
        subSlice = createFQ(rewriter, subSlice, inputFQ);
    }

    SmallVector<mlir::Value> subSlices;
    subSlices.push_back(input);
    subSlices.insert(subSlices.end(), nums, subSlice);
    auto concatOp = rewriter.create<IE::ConcatOp>(loc, subSlices, axis).output();
    if (inputFQ != nullptr) {
        concatOp = createFQ(rewriter, concatOp, inputFQ);
    }

    return concatOp;
}

}  // namespace IE
}  // namespace vpux

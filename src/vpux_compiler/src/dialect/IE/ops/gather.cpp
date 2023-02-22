//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {

mlir::FailureOr<int64_t> extractAxis(mlir::Location loc, IE::GatherOpAdaptor gather) {
    if (gather.axis() != nullptr) {
        auto axisConst = gather.axis().getDefiningOp<Const::DeclareOp>();
        if (axisConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for axis");
        }

        const auto axisContent = axisConst.content();
        if (!axisContent.isSplat()) {
            return errorAt(loc, "Axis value must be a scalar");
        }

        int64_t axisInd = axisContent.getSplatValue<int64_t>();

        if (axisInd < 0) {
            const auto inType = gather.input().getType().cast<mlir::ShapedType>();
            const auto inRank = inType.getRank();
            axisInd += inRank;
            VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Wrong Gather axis {0}", axisInd);
        }

        return axisInd;
    } else if (gather.axis_value().hasValue()) {
        return gather.axis_value().getValue();
    } else {
        return errorAt(loc, "Axis was not provided");
    }
}

}  // namespace

mlir::LogicalResult vpux::IE::GatherOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::GatherOpAdaptor gather(operands, attrs);
    if (mlir::failed(gather.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = gather.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();
    const auto indicesShape = gather.indices().getType().cast<mlir::ShapedType>().getShape();

    const auto axis = extractAxis(loc, gather);
    if (mlir::failed(axis)) {
        return mlir::failure();
    }

    SmallVector<int64_t> outShape;

    // calculate output shape
    int64_t batchDims = gather.batch_dims();
    int64_t axisVal = checked_cast<int64_t>(*axis);
    int64_t outRank = inputShape.size() + indicesShape.size() - 1 - batchDims;
    int64_t indicesRank = indicesShape.size();
    int64_t i = 0;

    for (; i < batchDims; i++) {
        outShape.push_back(inputShape[i] & indicesShape[i]);
    }
    for (; i < axisVal; i++) {
        outShape.push_back(inputShape[i]);
    }
    for (; i < axisVal + indicesRank - batchDims; i++) {
        outShape.push_back(indicesShape[batchDims - axisVal + i]);
    }
    for (; i < outRank; i++) {
        outShape.push_back(inputShape[batchDims + 1 - indicesRank + i]);
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    using mlir::OpRewritePattern<IE::GatherOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const {
    auto axis = gatherOp.axis();
    if (axis == nullptr) {
        return mlir::failure();
    }

    auto axisConst = gatherOp.axis().getDefiningOp<Const::DeclareOp>();
    if (axisConst == nullptr) {
        return mlir::failure();
    }

    const auto axisContent = axisConst.content();
    if (!axisContent.isSplat()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::GatherOp>(gatherOp, gatherOp.getType(), gatherOp.input(), gatherOp.indices(),
                                              nullptr, rewriter.getI64IntegerAttr(axisContent.getSplatValue<int64_t>()),
                                              gatherOp.batch_dims());
    return mlir::success();
}

}  // namespace

void vpux::IE::GatherOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}

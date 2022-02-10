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

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/dialect/IE/utils/to_ngraph.hpp"

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
    } else if (gather.axis_value() != nullptr) {
        return gather.axis_value().getValue().getSExtValue();
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
    int64_t batchDims = gather.batch_dims().getValue().getSExtValue();
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
    patterns.insert<ConvertConstToAttr>(context);
}

std::unique_ptr<ngraph::Node> vpux::IE::GatherOp::toNgraph(ngraph::OutputVector &outputs)
{
    return std::make_unique<opset_latest::Gather>(outputs.at(0), outputs.at(1), outputs.at(2));
}

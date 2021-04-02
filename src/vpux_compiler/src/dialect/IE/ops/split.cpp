//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//
// getAxis
//

namespace {

Dim normalizeAxis(IE::SplitOpAdaptor split) {
    VPUX_THROW_UNLESS(split.axis_value() != nullptr, "Got non constant axis");

    const auto inType = split.input().getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();

    auto axisInd = split.axis_value().getSInt();

    // Negative value means counting dimension from the end
    if (axisInd < 0) {
        axisInd += inRank;
    }

    VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Split axis '{0}', out of range '{1}'", axisInd,
                      inRank);

    return Dim(axisInd);
}

}  // namespace

Dim vpux::IE::SplitOp::getAxis() {
    return normalizeAxis(*this);
}

//
// inferReturnTypeComponents
//

namespace {

mlir::FailureOr<Dim> extractAxis(mlir::Location loc, IE::SplitOpAdaptor split) {
    if (split.axis() != nullptr) {
        auto axisConst = split.axis().getDefiningOp<ConstantInterface>();

        if (axisConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for axis");
        }

        if (axisConst.getContent().size() != 1) {
            return errorAt(loc, "Axis value must be a scalar");
        }

        const auto inType = split.input().getType().cast<mlir::ShapedType>();
        const auto inRank = inType.getRank();

        auto axisInd = axisConst.getContent().getValues<int64_t>()[0];

        // Negative value means counting dimension from the end
        if (axisInd < 0) {
            axisInd += inRank;
        }

        VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Split axis '{0}', out of range '{1}'", axisInd,
                          inRank);

        return Dim(axisInd);
    } else if (split.axis_value() != nullptr) {
        return normalizeAxis(split);
    } else {
        return errorAt(loc, "Axis was not provided");
    }
}

}  // namespace

mlir::LogicalResult vpux::IE::SplitOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::SplitOpAdaptor split(operands, attrs);
    if (mlir::failed(split.verify(loc))) {
        return mlir::failure();
    }

    const auto axis = extractAxis(loc, split);
    if (mlir::failed(axis)) {
        return mlir::failure();
    }

    const auto num_splits = split.num_splits().getInt();

    auto outShape = getShape(split.input()).toValues();
    if ((outShape[*axis] < num_splits) || (outShape[*axis] % num_splits != 0)) {
        return errorAt(loc, "Unsupported num_splits parameter");
    }
    outShape[*axis] /= num_splits;

    const auto elemType = split.input().getType().cast<mlir::ShapedType>().getElementType();
    for (int i = 0; i < num_splits; ++i) {
        inferredReturnShapes.emplace_back(outShape.raw(), elemType);
    }

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    using mlir::OpRewritePattern<IE::SplitOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp splitOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::SplitOp splitOp, mlir::PatternRewriter& rewriter) const {
    if (splitOp.axis_value().hasValue()) {
        return mlir::failure();
    }

    const auto axis = extractAxis(splitOp.getLoc(), splitOp);
    if (mlir::failed(axis)) {
        return mlir::failure();
    }

    const auto axisAttr = getSInt32Attr(splitOp.getContext(), axis->ind());
    rewriter.replaceOpWithNewOp<IE::SplitOp>(splitOp, splitOp.input(), nullptr, splitOp.num_splitsAttr(), axisAttr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::SplitOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                    mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}

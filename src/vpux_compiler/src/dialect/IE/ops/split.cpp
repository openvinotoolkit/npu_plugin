//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

//
// inferReturnTypeComponents
//

namespace {

Dim normalizeAxis(IE::SplitOpAdaptor split) {
    VPUX_THROW_UNLESS(split.axis_value().hasValue(), "Got non constant axis");

    const auto inType = split.input().getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();

    auto axisInd = split.axis_value().getValue();

    // Negative value means counting dimension from the end
    if (axisInd < 0) {
        axisInd += inRank;
    }

    VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Split axis '{0}', out of range '{1}'", axisInd,
                      inRank);

    return Dim(axisInd);
}

mlir::FailureOr<Dim> extractAxis(mlir::Location loc, IE::SplitOpAdaptor split) {
    if (split.axis() != nullptr) {
        auto axisConst = split.axis().getDefiningOp<Const::DeclareOp>();
        if (axisConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for axis");
        }

        const auto axisContent = axisConst.content();
        if (!axisContent.isSplat()) {
            return errorAt(loc, "Axis value must be a scalar");
        }

        const auto inType = split.input().getType().cast<mlir::ShapedType>();
        const auto inRank = inType.getRank();

        auto axisInd = axisContent.getSplatValue<int64_t>();

        // Negative value means counting dimension from the end
        if (axisInd < 0) {
            axisInd += inRank;
        }

        VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Split axis '{0}', out of range '{1}'", axisInd,
                          inRank);

        return Dim(axisInd);
    } else if (split.axis_value().hasValue()) {
        return normalizeAxis(split);
    } else {
        return errorAt(loc, "Axis was not provided");
    }
}

}  // namespace

mlir::LogicalResult vpux::IE::SplitOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SplitOpAdaptor split(operands, attrs);
    if (mlir::failed(split.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = split.input().getType().cast<mlir::RankedTensorType>();

    const auto axis = extractAxis(loc, split);
    if (mlir::failed(axis)) {
        return mlir::failure();
    }

    const auto num_splits = split.num_splits();

    auto outShape = inType.cast<vpux::NDTypeInterface>().getShape().toValues();
    if ((outShape[*axis] < num_splits) || (outShape[*axis] % num_splits != 0)) {
        return errorAt(loc, "Unsupported num_splits parameter");
    }
    outShape[*axis] /= num_splits;

    const auto elemType = inType.getElementType();
    const auto outDesc = IE::getTensorAttr(inType);

    for (int i = 0; i < num_splits; ++i) {
        inferredReturnShapes.emplace_back(outShape.raw(), elemType, outDesc);
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

    const auto axisAttr = getIntAttr(splitOp.getContext(), axis->ind());
    rewriter.replaceOpWithNewOp<IE::SplitOp>(splitOp, splitOp.input(), nullptr, splitOp.num_splitsAttr(), axisAttr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::SplitOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}

//
// inferElemTypeInfo
//

void vpux::IE::SplitOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    auto outputElemType = info.getInput(0);

    // Do not propagate element type down in per channel case.
    if (outputElemType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>() == nullptr) {
        for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
            info.setOutput(outputInd, outputElemType);
        }
    }
}

void vpux::IE::SplitOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    const auto outputElemType = info.getOutput(0);

    if (outputElemType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>() != nullptr) {
        // E#31029: implement propagate type up for per channel, currently it leads to failures in later passes.
        return;
    }

    for (size_t inputInd = 0; inputInd < info.getNumInputs(); ++inputInd) {
        info.setInput(inputInd, outputElemType);
    }
}

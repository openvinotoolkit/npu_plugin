//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

#include <ngraph/validation_util.hpp>

using namespace vpux;

namespace {

struct StridedSliceInputData final {
    SmallVector<int64_t> begins;
    SmallVector<int64_t> ends;
    SmallVector<int64_t> strides;
};

mlir::FailureOr<StridedSliceInputData> extractData(mlir::Location loc, IE::StridedSliceOpAdaptor stridedSlice) {
    if (stridedSlice.getBegins() != nullptr) {
        auto begins = IE::constInputToData(loc, stridedSlice.getBegins());
        auto ends = IE::constInputToData(loc, stridedSlice.getEnds());
        auto strides = IE::constInputToData(loc, stridedSlice.getStrides());

        if (mlir::failed(begins) || mlir::failed(ends) || mlir::failed(strides)) {
            return mlir::failure();
        }

        return StridedSliceInputData{begins.value(), ends.value(), strides.value()};
    }

    if (stridedSlice.getBeginsAttr().has_value()) {
        auto begins = parseIntArrayAttr<int64_t>(stridedSlice.getBeginsAttr().value());
        auto ends = parseIntArrayAttr<int64_t>(stridedSlice.getEndsAttr().value());
        auto strides = parseIntArrayAttr<int64_t>(stridedSlice.getStridesAttr().value());

        return StridedSliceInputData{std::move(begins), std::move(ends), std::move(strides)};
    }

    return mlir::failure();
}

}  // namespace

mlir::LogicalResult vpux::IE::StridedSliceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::StridedSliceOpAdaptor slice(operands, attrs);
    if (mlir::failed(slice.verify(loc))) {
        return mlir::failure();
    }

    const auto getAxisSetArr = [](mlir::ArrayAttr attr) {
        ov::AxisSet axis_set;

        const auto arr = parseIntArrayAttr<int64_t>(attr);
        for (const auto& p : arr | indexed) {
            if (p.value() == 1) {
                axis_set.emplace(p.index());
            }
        }

        return axis_set;
    };

    const auto inDataType = slice.getInput().getType().cast<mlir::ShapedType>();
    const auto inDataShape = inDataType.getShape();

    const auto inputData = extractData(loc, slice);
    if (mlir::failed(inputData)) {
        return mlir::failure();
    }

    const auto begins = to_std_vector(inputData.value().begins);
    const auto ends = to_std_vector(inputData.value().ends);
    const auto strides = to_std_vector(inputData.value().strides);

    const auto beginMask = getAxisSetArr(slice.getBeginMask());
    const auto endMask = getAxisSetArr(slice.getEndMask());
    const auto newAxisMask = getAxisSetArr(slice.getNewAxisMask());
    const auto shrinkAxisMask = getAxisSetArr(slice.getShrinkAxisMask());
    const auto ellipsisMask = getAxisSetArr(slice.getEllipsisMask());

    const auto outputShape =
            ngraph::infer_slice_shape(EmptyNode::instance(), ov::Shape(inDataShape.begin(), inDataShape.end()), begins,
                                      ends, strides, beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask);

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    auto inType = slice.getInput().getType().cast<NDTypeInterface>();
    const auto outType = inType.changeShape(Shape(shapeI64)).cast<mlir::RankedTensorType>();

    inferredReturnShapes.emplace_back(outType.getShape(), outType.getElementType(), outType.getEncoding());

    return mlir::success();
}

bool vpux::IE::StridedSliceOp::isSimplified() {
    auto isZero = [](auto val) {
        return val == 0;
    };
    auto isPositive = [](auto val) {
        return val >= 0;
    };

    return (llvm::all_of(parseIntArrayAttr<int64_t>(getNewAxisMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getShrinkAxisMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getEllipsisMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getBeginMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getEndMask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getBeginsAttr().value()), isPositive) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getEndsAttr().value()), isPositive));
}

//
// fold
//

mlir::OpFoldResult vpux::IE::StridedSliceOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    // TODO E#22568: attempt const folding but only if slice isSimplified()

    return nullptr;
}

//
// ComposeStridedSlice
//

namespace {

class ComposeStridedSlice final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ComposeStridedSlice::matchAndRewrite(IE::StridedSliceOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto producerSliceOp = origOp.getInput().getDefiningOp<IE::StridedSliceOp>();
    if (producerSliceOp == nullptr) {
        return mlir::failure();
    }

    if (!(origOp.isSimplified() && producerSliceOp.isSimplified())) {
        return mlir::failure();
    }

    const auto firstBegin = parseIntArrayAttr<int64_t>(producerSliceOp.getBeginsAttr().value());
    const auto nextBegin = parseIntArrayAttr<int64_t>(origOp.getBeginsAttr().value());
    auto resultBegin = SmallVector<int64_t>(nextBegin.size());

    const auto firstEnd = parseIntArrayAttr<int64_t>(producerSliceOp.getEndsAttr().value());
    const auto nextEnd = parseIntArrayAttr<int64_t>(origOp.getEndsAttr().value());
    auto resultEnd = SmallVector<int64_t>(nextEnd.size());

    const auto firstStride = parseIntArrayAttr<int64_t>(producerSliceOp.getStridesAttr().value());
    const auto nextStride = parseIntArrayAttr<int64_t>(origOp.getStridesAttr().value());
    auto resultStride = SmallVector<int64_t>(nextStride.size());

    for (auto i : irange(firstBegin.size())) {
        resultBegin[i] = firstBegin[i] + nextBegin[i] * firstStride[i];
        resultEnd[i] = firstBegin[i] + nextEnd[i] * firstStride[i];
        resultStride[i] = firstStride[i] * nextStride[i];
    }

    // Stride on more than 2 axis is not supported
    const auto greaterThanOne = [](auto ele) {
        return ele > 1;
    };
    auto stridesGreaterThanOne = llvm::count_if(resultStride, greaterThanOne);
    if (stridesGreaterThanOne > 2) {
        return mlir::failure();
    }

    const auto beginsAttr = getIntArrayAttr(getContext(), resultBegin);
    const auto endsAttr = getIntArrayAttr(getContext(), resultEnd);
    const auto stridesAttr = getIntArrayAttr(getContext(), resultStride);

    const auto fusedLoc =
            mlir::FusedLoc::get(producerSliceOp->getLoc().getContext(), {producerSliceOp->getLoc(), origOp->getLoc()});
    const auto newOp = rewriter.create<IE::StridedSliceOp>(
            fusedLoc, producerSliceOp.getInput(), origOp.getBegins(), origOp.getEnds(), origOp.getStrides(), beginsAttr,
            endsAttr, stridesAttr, origOp.getBeginMask(), origOp.getEndMask(), origOp.getNewAxisMask(),
            origOp.getShrinkAxisMask(), origOp.getEllipsisMask());
    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// ConvertConstToAttr
//

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    using mlir::OpRewritePattern<IE::StridedSliceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp stridedSliceOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::StridedSliceOp slice,
                                                        mlir::PatternRewriter& rewriter) const {
    if (!slice.getBegins() || !slice.getEnds() || !slice.getStrides()) {
        return mlir::failure();
    }

    const auto inputData = extractData(slice.getLoc(), slice);
    if (mlir::failed(inputData)) {
        return mlir::failure();
    }

    const auto beginsAttr = getIntArrayAttr(getContext(), inputData.value().begins);
    const auto endsAttr = getIntArrayAttr(getContext(), inputData.value().ends);
    const auto stridesAttr = getIntArrayAttr(getContext(), inputData.value().strides);

    rewriter.replaceOpWithNewOp<IE::StridedSliceOp>(
            slice, slice.getInput(), nullptr, nullptr, nullptr, beginsAttr, endsAttr, stridesAttr, slice.getBeginMask(),
            slice.getEndMask(), slice.getNewAxisMask(), slice.getShrinkAxisMask(), slice.getEllipsisMask());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::StridedSliceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                           mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
    patterns.add<ComposeStridedSlice>(context);
}

//
// inferElemTypeInfo
//

void vpux::IE::StridedSliceOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    // E#84659: implement propagate type up for per channel, currently it leads to failures in later passes.
    propagateElementTypeDown(info);
}

void vpux::IE::StridedSliceOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    // E#84659: implement propagate type up for per channel, currently it leads to failures in later passes.
    propagateElementTypeUp(info);
}

//
// verify
//

mlir::LogicalResult vpux::IE::StridedSliceOp::verify() {
    auto beginsAttr = getBeginsAttr();
    auto endsAttr = getEndsAttr();
    auto stridesAttr = getStridesAttr();

    if (beginsAttr.has_value() && endsAttr.has_value()) {
        if (beginsAttr.value().size() != endsAttr.value().size()) {
            return errorAt(*this, "lower bounds and Upper bounds needs to have same number of values");
        }
    }

    if (beginsAttr.has_value() && stridesAttr.has_value()) {
        if (beginsAttr.value().size() != endsAttr.value().size()) {
            return errorAt(*this, "lower bounds and strides needs to have same number of values");
        }
    }

    if (endsAttr.has_value() && stridesAttr.has_value()) {
        if (beginsAttr.value().size() != endsAttr.value().size()) {
            return errorAt(*this, "Upper bounds and strides needs to have same number of values");
        }
    }

    return mlir::success();
}

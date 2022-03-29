//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <ngraph/slice_plan.hpp>
#include <ngraph/validation_util.hpp>
#include <vpux/utils/core/logger.hpp>

using namespace vpux;

namespace {

struct StridedSliceInputData final {
    SmallVector<int64_t> begins;
    SmallVector<int64_t> ends;
    SmallVector<int64_t> strides;
};

mlir::FailureOr<StridedSliceInputData> extractData(mlir::Location loc, IE::StridedSliceOpAdaptor stridedSlice) {
    if (stridedSlice.begins() != nullptr) {
        auto begins = IE::constInputToData(loc, stridedSlice.begins());
        auto ends = IE::constInputToData(loc, stridedSlice.ends());
        auto strides = IE::constInputToData(loc, stridedSlice.strides());

        if (mlir::failed(begins) || mlir::failed(ends) || mlir::failed(strides)) {
            return mlir::failure();
        }

        return StridedSliceInputData{begins.getValue(), ends.getValue(), strides.getValue()};
    }

    if (stridedSlice.begins_attr() != nullptr) {
        auto begins = parseIntArrayAttr<int64_t>(stridedSlice.begins_attr());
        auto ends = parseIntArrayAttr<int64_t>(stridedSlice.ends_attr());
        auto strides = parseIntArrayAttr<int64_t>(stridedSlice.strides_attr());

        return StridedSliceInputData{std::move(begins), std::move(ends), std::move(strides)};
    }

    return mlir::failure();
}

}  // namespace

mlir::LogicalResult vpux::IE::StridedSliceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::StridedSliceOpAdaptor slice(operands, attrs);
    if (mlir::failed(slice.verify(loc))) {
        return mlir::failure();
    }

    const auto getAxisSetArr = [](mlir::ArrayAttr attr) {
        ngraph::AxisSet axis_set;

        const auto arr = parseIntArrayAttr<int64_t>(attr);
        for (const auto& p : arr | indexed) {
            if (p.value() == 1) {
                axis_set.emplace(p.index());
            }
        }

        return axis_set;
    };

    const auto inDataType = slice.input().getType().cast<mlir::ShapedType>();
    const auto inDataShape = inDataType.getShape();

    const auto inputData = extractData(loc, slice);
    if (mlir::failed(inputData)) {
        return mlir::failure();
    }

    const auto begins = to_std_vector(inputData.getValue().begins);
    const auto ends = to_std_vector(inputData.getValue().ends);
    const auto strides = to_std_vector(inputData.getValue().strides);

    const auto beginMask = getAxisSetArr(slice.begin_mask());
    const auto endMask = getAxisSetArr(slice.end_mask());
    const auto newAxisMask = getAxisSetArr(slice.new_axis_mask());
    const auto shrinkAxisMask = getAxisSetArr(slice.shrink_axis_mask());
    const auto ellipsisMask = getAxisSetArr(slice.ellipsis_mask());

    const auto outputShape =
            ngraph::infer_slice_shape(nullptr, ngraph::Shape(inDataShape.begin(), inDataShape.end()), begins, ends,
                                      strides, beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask);

    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    inferredReturnShapes.emplace_back(shapeI64, inDataType.getElementType());

    return mlir::success();
}

bool vpux::IE::StridedSliceOp::isSimplified() {
    auto isZero = [](auto val) {
        return val == 0;
    };
    auto isPositive = [](auto val) {
        return val >= 0;
    };

    return (llvm::all_of(parseIntArrayAttr<int64_t>(new_axis_mask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(shrink_axis_mask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(ellipsis_mask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(begin_mask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(end_mask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(begins_attr().getValue()), isPositive) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(ends_attr().getValue()), isPositive));
}

//
// fold
//

mlir::OpFoldResult vpux::IE::StridedSliceOp::fold(ArrayRef<mlir::Attribute> /*operands*/) {
    if (input().getType() == output().getType()) {
        return input();
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
    auto producerSliceOp = origOp.input().getDefiningOp<IE::StridedSliceOp>();
    if (producerSliceOp == nullptr) {
        return mlir::failure();
    }

    if (!(origOp.isSimplified() && producerSliceOp.isSimplified())) {
        return mlir::failure();
    }

    const auto firstBegin = parseIntArrayAttr<int64_t>(producerSliceOp.begins_attr().getValue());
    const auto nextBegin = parseIntArrayAttr<int64_t>(origOp.begins_attr().getValue());
    auto resultBegin = SmallVector<int64_t>(nextBegin.size());

    const auto firstEnd = parseIntArrayAttr<int64_t>(producerSliceOp.ends_attr().getValue());
    const auto nextEnd = parseIntArrayAttr<int64_t>(origOp.ends_attr().getValue());
    auto resultEnd = SmallVector<int64_t>(nextEnd.size());

    const auto firstStride = parseIntArrayAttr<int64_t>(producerSliceOp.strides_attr().getValue());
    const auto nextStride = parseIntArrayAttr<int64_t>(origOp.strides_attr().getValue());
    auto resultStride = SmallVector<int64_t>(nextStride.size());

    for (auto i : irange(firstBegin.size())) {
        resultBegin[i] = firstBegin[i] + nextBegin[i] * firstStride[i];
        resultEnd[i] = firstBegin[i] + nextEnd[i] * firstStride[i];
        resultStride[i] = firstStride[i] * nextStride[i];
    }

    const auto beginsAttr = getIntArrayAttr(getContext(), resultBegin);
    const auto endsAttr = getIntArrayAttr(getContext(), resultEnd);
    const auto stridesAttr = getIntArrayAttr(getContext(), resultStride);

    const auto fusedLoc =
            mlir::FusedLoc::get(producerSliceOp->getLoc().getContext(), {producerSliceOp->getLoc(), origOp->getLoc()});
    const auto newOp = rewriter.create<IE::StridedSliceOp>(
            fusedLoc, producerSliceOp.input(), origOp.begins(), origOp.ends(), origOp.strides(), beginsAttr, endsAttr,
            stridesAttr, origOp.begin_mask(), origOp.end_mask(), origOp.new_axis_mask(), origOp.shrink_axis_mask(),
            origOp.ellipsis_mask());
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
    if (!slice.begins() || !slice.ends() || !slice.strides()) {
        return mlir::failure();
    }

    const auto inputData = extractData(slice.getLoc(), slice);
    if (mlir::failed(inputData)) {
        return mlir::failure();
    }

    const auto beginsAttr = getIntArrayAttr(getContext(), inputData.getValue().begins);
    const auto endsAttr = getIntArrayAttr(getContext(), inputData.getValue().ends);
    const auto stridesAttr = getIntArrayAttr(getContext(), inputData.getValue().strides);

    rewriter.replaceOpWithNewOp<IE::StridedSliceOp>(
            slice, slice.input(), nullptr, nullptr, nullptr, beginsAttr, endsAttr, stridesAttr, slice.begin_mask(),
            slice.end_mask(), slice.new_axis_mask(), slice.shrink_axis_mask(), slice.ellipsis_mask());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::StridedSliceOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                           mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
    patterns.insert<ComposeStridedSlice>(context);
}

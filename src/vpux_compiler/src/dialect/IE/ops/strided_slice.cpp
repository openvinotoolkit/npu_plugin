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
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <ngraph/slice_plan.hpp>
#include <ngraph/validation_util.hpp>
#include <vpux/utils/core/logger.hpp>

using namespace vpux;

namespace {

mlir::FailureOr<llvm::SmallVector<int64_t>> constInputToData(mlir::Location loc, mlir::Value input) {
    auto constantData = input.getDefiningOp<ConstantInterface>();

    if (constantData == nullptr) {
        return errorAt(loc, "Only constant input is supported");
    }

    return to_small_vector(constantData.getContent().getValues<int64_t>());
}

struct StridedSliceInputData {
    llvm::SmallVector<int64_t> begins;
    llvm::SmallVector<int64_t> ends;
    llvm::SmallVector<int64_t> strides;
};

mlir::FailureOr<StridedSliceInputData> extractData(mlir::Location loc, IE::StridedSliceOpAdaptor stridedSlice) {
    if (stridedSlice.begins() != nullptr) {
        auto begins = constInputToData(loc, stridedSlice.begins());
        auto ends = constInputToData(loc, stridedSlice.ends());
        auto strides = constInputToData(loc, stridedSlice.strides());
        if (mlir::failed(begins) || mlir::failed(ends) || mlir::failed(strides)) {
            return mlir::failure();
        }
        return StridedSliceInputData{begins.getValue(), ends.getValue(), strides.getValue()};
    } else if (stridedSlice.begins_attr() != nullptr) {
        auto begins = to_small_vector(parseIntArrayAttr(stridedSlice.begins_attr()));
        auto ends = to_small_vector(parseIntArrayAttr(stridedSlice.ends_attr()));
        auto strides = to_small_vector(parseIntArrayAttr(stridedSlice.strides_attr()));
        return StridedSliceInputData{begins, ends, strides};
    } else {
        VPUX_THROW("StridedSlice operation is invalid");
    }
}

}  // namespace

mlir::LogicalResult vpux::IE::StridedSliceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::StridedSliceOpAdaptor slice(operands, attrs);
    if (mlir::failed(slice.verify(loc))) {
        return mlir::failure();
    }

    const auto getAxisSetArr = [](mlir::ArrayAttr attr) {
        ngraph::AxisSet axis_set;

        const auto arr = parseIntArrayAttr(attr);
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
    auto log = vpux::Logger::global();
    inferredReturnShapes.emplace_back(shapeI64, inDataType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    using mlir::OpRewritePattern<IE::StridedSliceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp stridedSliceOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::StridedSliceOp stridedSliceOp,
                                                        mlir::PatternRewriter& rewriter) const {
    if (!stridedSliceOp.begins() || !stridedSliceOp.ends() || !stridedSliceOp.strides()) {
        return mlir::failure();
    }

    const auto inputData = extractData(stridedSliceOp.getLoc(), stridedSliceOp);
    if (mlir::failed(inputData)) {
        return mlir::failure();
    }

    const auto ctx = stridedSliceOp.getContext();
    const auto beginsAttr = getInt64ArrayAttr(ctx, inputData.getValue().begins);
    const auto endsAttr = getInt64ArrayAttr(ctx, inputData.getValue().ends);
    const auto stridesAttr = getInt64ArrayAttr(ctx, inputData.getValue().strides);

    rewriter.replaceOpWithNewOp<IE::StridedSliceOp>(stridedSliceOp, stridedSliceOp.input(), nullptr, nullptr, nullptr,
                                                    beginsAttr, endsAttr, stridesAttr, stridedSliceOp.begin_mask(),
                                                    stridedSliceOp.end_mask(), stridedSliceOp.new_axis_mask(),
                                                    stridedSliceOp.shrink_axis_mask(), stridedSliceOp.ellipsis_mask());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::StridedSliceOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns,
                                                           mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}

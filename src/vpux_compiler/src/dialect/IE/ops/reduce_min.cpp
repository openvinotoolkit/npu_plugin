//
// Created by liuhao on 2021/9/24.
//
#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// getAxes
//

namespace {

mlir::FailureOr<SmallVector<int64_t>> getAxes(IE::ReduceMinOpAdaptor reduceMin, mlir::Location loc) {
    if (reduceMin.axes() == nullptr) {
        return errorAt(loc, "Missed axes representation");
    }

    auto axesConst = reduceMin.axes().getDefiningOp<Const::DeclareOp>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant axes are supported");
    }

    const auto axesContent = axesConst.content();
    auto axes = to_small_vector(axesContent.getValues<int64_t>());

    const auto inType = reduceMin.input().getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inRank;
        }
    }
    std::sort(axes.begin(), axes.end());

    return axes;
}

}  // namespace

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ReduceMinOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        mlir::SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ReduceMinOpAdaptor reduceMin(operands, attrs);
    if (mlir::failed(reduceMin.verify(loc))) {
        return mlir::failure();
    }

    const auto axes = getAxes(reduceMin, loc);
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto inType = reduceMin.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    const auto keepDims = reduceMin.keep_dims().getValue();

    SmallVector<int64_t> outShape;
    size_t axesInd = 0;
    for (auto inInd : irange(inType.getRank())) {
        if (axesInd < axes->size() && inInd == axes->begin()[axesInd]) {
            axesInd++;
            if (keepDims) {
                outShape.push_back(1);
            }
            continue;
        }
        outShape.push_back(inShape[inInd]);
    }

    inferredReturnShapes.emplace_back(makeArrayRef(outShape), inType.getElementType());
    return mlir::success();
}
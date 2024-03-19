//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

#include <ngraph/validation_util.hpp>
#include <vpux/utils/core/logger.hpp>

using namespace vpux;

namespace {

struct StridedSliceInputData final {
    SmallVector<int64_t> begins;
    SmallVector<int64_t> ends;
    SmallVector<int64_t> strides;
};

mlir::FailureOr<StridedSliceInputData> extractData(VPU::StridedSliceOpAdaptor stridedSlice) {
    if (stridedSlice.getBeginsAttr()) {
        auto begins = parseIntArrayAttr<int64_t>(stridedSlice.getBeginsAttr());
        auto ends = parseIntArrayAttr<int64_t>(stridedSlice.getEndsAttr());
        auto strides = parseIntArrayAttr<int64_t>(stridedSlice.getStridesAttr().value());

        return StridedSliceInputData{std::move(begins), std::move(ends), std::move(strides)};
    }

    return mlir::failure();
}

}  // namespace
// TODO: E-90249 Extend the infer type logic for StridedSlice to support different input / output ranks
mlir::LogicalResult vpux::VPU::StridedSliceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::StridedSliceOpAdaptor slice(operands, attrs);
    if (mlir::failed(slice.verify(loc))) {
        return mlir::failure();
    }

    const auto getAxisSetArr = [](mlir::ArrayAttr attr) {
        ov::AxisSet axisSet;

        const auto arr = parseIntArrayAttr<int64_t>(attr);
        for (const auto& p : arr | indexed) {
            if (p.value() == 1) {
                axisSet.emplace(p.index());
            }
        }

        return axisSet;
    };

    const auto inDataType = slice.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inDataShape = inDataType.getShape();

    const auto inputData = extractData(slice);

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

    const auto newShape = Shape(shapeI64);
    auto outType = inDataType.changeShape(newShape);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

bool vpux::VPU::StridedSliceOp::isSimplified() {
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
            llvm::all_of(parseIntArrayAttr<int64_t>(getBeginsAttr()), isPositive) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(getEndsAttr()), isPositive));
}

InputTiling vpux::VPU::StridedSliceOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    const auto inShape = getShape(getInput());
    const auto begins = Shape(parseIntArrayAttr<int64_t>(getBeginsAttrAttr()));
    const auto strides = Shape(parseIntArrayAttr<int64_t>(getStridesAttrAttr()));
    const auto outOrder = DimsOrder::fromValue(getOutput());
    auto curTile = outputTile;
    for (auto ind : irange(inShape.size())) {
        auto idx = outOrder.dimAt(ind);
        curTile.shape[idx] = outputTile.shape[idx] * strides[idx];
        curTile.offsets[idx] = outputTile.offsets[idx] * strides[idx] + begins[idx];
        curTile.axis[idx] = outputTile.axis[idx];
    }
    return TilingInfo{curTile};
}

void vpux::VPU::StridedSliceOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    const auto inShape = getShape(getInput());
    auto ends = parseIntArrayAttr<int64_t>(getEndsAttr());
    auto begins = parseIntArrayAttr<int64_t>(getBeginsAttr());
    for (auto ind : irange(inShape.size())) {
        begins[ind] = 0;
        ends[ind] = inputTiling.tiles[0].shape[Dim(ind)];
    }
    const auto newEndsAttr = getIntArrayAttr(getContext(), ends);
    const auto newBeginsAttr = getIntArrayAttr(getContext(), begins);
    setEndsAttrAttr(newEndsAttr);
    setBeginsAttrAttr(newBeginsAttr);
}

mlir::FailureOr<OutputTiling> vpux::VPU::StridedSliceOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

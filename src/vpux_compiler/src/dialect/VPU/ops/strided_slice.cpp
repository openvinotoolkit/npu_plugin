//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

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

mlir::FailureOr<StridedSliceInputData> extractData(VPU::StridedSliceOpAdaptor stridedSlice) {
    if (stridedSlice.begins_attr()) {
        auto begins = parseIntArrayAttr<int64_t>(stridedSlice.begins_attr());
        auto ends = parseIntArrayAttr<int64_t>(stridedSlice.ends_attr());
        auto strides = parseIntArrayAttr<int64_t>(stridedSlice.strides_attr().getValue());

        return StridedSliceInputData{std::move(begins), std::move(ends), std::move(strides)};
    }

    return mlir::failure();
}

}  // namespace

mlir::LogicalResult vpux::VPU::StridedSliceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::StridedSliceOpAdaptor slice(operands, attrs);
    if (mlir::failed(slice.verify(loc))) {
        return mlir::failure();
    }

    const auto getAxisSetArr = [](mlir::ArrayAttr attr) {
        ngraph::AxisSet axisSet;

        const auto arr = parseIntArrayAttr<int64_t>(attr);
        for (const auto& p : arr | indexed) {
            if (p.value() == 1) {
                axisSet.emplace(p.index());
            }
        }

        return axisSet;
    };

    const auto inDataType = slice.input().getType().cast<vpux::NDTypeInterface>();
    const auto inDataShape = inDataType.getShape();

    const auto inputData = extractData(slice);

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

    const auto newShape = Shape(shapeI64);
    auto outType = inDataType.changeShape(newShape);
    outType = outType.changeDimsOrder(DimsOrder::fromNumDims(newShape.size()));
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

    return (llvm::all_of(parseIntArrayAttr<int64_t>(new_axis_mask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(shrink_axis_mask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(ellipsis_mask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(begin_mask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(end_mask()), isZero) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(begins_attr()), isPositive) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(ends_attr()), isPositive));
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::StridedSliceOp::serialize(EMU::BlobWriter& writer) {
    auto attrToVector = [&](mlir::ArrayAttr attr) {
        return to_std_vector(parseIntArrayAttr<uint32_t>(attr));
    };

    const auto beginsVec = attrToVector(begins_attrAttr());
    const auto endsVec = attrToVector(ends_attrAttr());
    const auto stridesVec = attrToVector(strides_attrAttr());

    const auto paramsOff = MVCNN::CreateStridedSliceParamsDirect(writer, &beginsVec, &endsVec, &stridesVec);

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_StridedSliceParams});
}

InputTiling vpux::VPU::StridedSliceOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    const auto inShape = getShape(input());
    const auto begins = Shape(parseIntArrayAttr<int64_t>(begins_attrAttr()));
    const auto strides = Shape(parseIntArrayAttr<int64_t>(strides_attrAttr()));
    const auto outOrder = DimsOrder::fromValue(output());
    auto curTile = outputTile;
    for (auto ind : irange(inShape.size())) {
        auto idx = outOrder.dimAt(ind);
        curTile.shape[idx] = (outputTile.shape[idx] * strides[idx] - (strides[idx] - 1));
        curTile.offsets[idx] = outputTile.offsets[idx] * strides[idx] + begins[idx];
        curTile.axis[idx] = outputTile.axis[idx];
    }
    return TilingInfo{curTile};
}

void vpux::VPU::StridedSliceOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    const auto inShape = getShape(input());
    auto ends = parseIntArrayAttr<int64_t>(ends_attr());
    auto begins = parseIntArrayAttr<int64_t>(begins_attr());
    for (auto ind : irange(inShape.size())) {
        begins[ind] = 0;
        ends[ind] = inputTiling.tiles[0].shape[Dim(ind)];
    }
    const auto newEndsAttr = getIntArrayAttr(getContext(), ends);
    const auto newBeginsAttr = getIntArrayAttr(getContext(), begins);
    ends_attrAttr(newEndsAttr);
    begins_attrAttr(newBeginsAttr);
}

OutputTiling vpux::VPU::StridedSliceOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

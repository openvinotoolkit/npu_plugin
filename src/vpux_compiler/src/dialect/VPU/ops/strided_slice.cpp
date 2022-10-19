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

mlir::FailureOr<StridedSliceInputData> extractData(mlir::Location loc, VPU::StridedSliceOpAdaptor stridedSlice) {
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
            llvm::all_of(parseIntArrayAttr<int64_t>(begins_attr().getValue()), isPositive) &&
            llvm::all_of(parseIntArrayAttr<int64_t>(ends_attr().getValue()), isPositive));
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

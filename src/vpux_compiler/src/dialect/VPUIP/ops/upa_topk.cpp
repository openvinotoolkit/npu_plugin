//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::TopKUPAOp::serialize(vpux::VPUIP::BlobWriter& writer) {
    auto axis = axisAttr().getInt();
    const auto inType = input().getType().cast<vpux::NDTypeInterface>();
    const auto inputDimension = inType.getRank();
    if (axis < 0) {
        axis = axis + inputDimension;
    }
    int32_t axis32 = checked_cast<int32_t>(axis);

    IE::TopKMode modeValue = mode();
    MVCNN::TopKMode modeCode = MVCNN::TopKMode::TopKMode_min;
    switch (modeValue) {
    case IE::TopKMode::MIN:
        modeCode = MVCNN::TopKMode::TopKMode_min;
        break;
    case IE::TopKMode::MAX:
        modeCode = MVCNN::TopKMode::TopKMode_max;
        break;
    }

    IE::TopKSortType sortValue = sort();
    MVCNN::TopKSort sortCode = MVCNN::TopKSort::TopKSort_value;
    switch (sortValue) {
    case IE::TopKSortType::SORT_VALUES:
        sortCode = MVCNN::TopKSort::TopKSort_value;
        break;
    case IE::TopKSortType::SORT_INDICES:
        sortCode = MVCNN::TopKSort::TopKSort_index;
        break;
    case IE::TopKSortType::NONE:
        sortCode = MVCNN::TopKSort::TopKSort_none;
        break;
    }

    MVCNN::TopKParamsBuilder builder(writer);
    builder.add_axis(axis32);
    builder.add_mode(modeCode);
    builder.add_sort(sortCode);
    builder.add_hasValues(true);
    builder.add_hasIndices(true);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_TopKParams});
}

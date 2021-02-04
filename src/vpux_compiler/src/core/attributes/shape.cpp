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

#include "vpux/compiler/core/attributes/shape.hpp"

#include "vpux/utils/core/error.hpp"

#include <algorithm>
#include <numeric>

#include <cassert>
#include <cstdint>

using namespace vpux;

//
// Shape utils
//

bool vpux::details::isDynamicDimValues(ArrayRef<int64_t> shape) {
    return std::any_of(shape.begin(), shape.end(), [](int64_t val) {
        return val <= 0;
    });
}

int64_t vpux::details::calcTotalShapeSize(ArrayRef<int64_t> shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t acc, int64_t d) {
        VPUX_THROW_UNLESS(d > 0, "Can't compute total shape size on dynamic shape");
        return acc * d;
    });
}

//
// Shape
//

ShapeRef vpux::getShape(mlir::ShapedType type) {
    return ShapeRef(type.getShape());
}

ShapeRef vpux::getShape(mlir::Value val) {
    auto type = val.getType().dyn_cast_or_null<mlir::ShapedType>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non ShapedType '{1}'", val, val.getType());
    return getShape(type);
}

//
// MemShape
//

MemShape vpux::getMemIndexND(int64_t memIndex1D, MemShapeRef memShape) {
    MemShape memIndexND(memShape.size());

    int64_t tempIndex1D = memIndex1D;
    for (size_t ind = 0; ind < memIndexND.size(); ++ind) {
        const auto md = MemDim(ind);
        const auto mdSize = memShape[md];

        memIndexND[md] = tempIndex1D % mdSize;
        tempIndex1D = (tempIndex1D - memIndexND[md]) / mdSize;

        VPUX_THROW_UNLESS(tempIndex1D >= 0, "Memory index 1D '{0}' is not compatible with memory shape '{1}'",
                          memIndex1D, memShape);
    }

    VPUX_THROW_UNLESS(tempIndex1D == 0, "Memory index 1D '{0}' is not compatible with memory shape '{1}'", memIndex1D,
                      memShape);

    return memIndexND;
}

int64_t vpux::getMemIndex1D(MemShapeRef memIndexND, MemShapeRef memShape) {
    VPUX_THROW_UNLESS(memIndexND.size() == memShape.size(),
                      "Memory index ND '{0}' is not compatible with memory shape '{1}'", memIndexND, memShape);

    int64_t memIndex1D = 0;

    int64_t tempSize = 1;
    for (size_t ind = 0; ind < memShape.size(); ++ind) {
        const auto md = MemDim(ind);
        const auto mdSize = memShape[md];
        const auto mdInd = memIndexND[md];

        VPUX_THROW_UNLESS(mdInd < mdSize, "Memory index ND '{0}' is not compatible with memory shape '{1}'", memIndexND,
                          memShape);

        memIndex1D += mdInd * tempSize;
        tempSize *= mdSize;
    }

    return memIndex1D;
}

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

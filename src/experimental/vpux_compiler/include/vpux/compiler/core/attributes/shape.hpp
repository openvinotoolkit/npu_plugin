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

#pragma once

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/dim_values.hpp"

#include <mlir/IR/BuiltinTypes.h>

#include <cstdint>

namespace vpux {

//
// Shape utils
//

namespace details {

bool isDynamicDimValues(ArrayRef<int64_t> shape);
int64_t calcTotalShapeSize(ArrayRef<int64_t> shape);

}  // namespace details

//
// ShapeImpl
//

namespace details {

template <class Base>
class ShapeTag : public Base {
public:
    using Base::Base;

public:
    bool isDynamic() const {
        return details::isDynamicDimValues(this->raw());
    }

    bool isStatic() const {
        return !isDynamic();
    }

public:
    int64_t totalSize() const {
        return details::calcTotalShapeSize(this->raw());
    }
};

}  // namespace details

//
// Shape
//

using Shape = details::DimValues<Dim, int64_t, details::ShapeTag>;
using ShapeRef = details::DimValuesRef<Dim, int64_t, details::ShapeTag>;

ShapeRef getShape(mlir::ShapedType type);

//
// MemShape
//

using MemShape = details::DimValues<MemDim, int64_t, details::ShapeTag>;
using MemShapeRef = details::DimValuesRef<MemDim, int64_t, details::ShapeTag>;

}  // namespace vpux

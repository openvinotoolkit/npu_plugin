//
// Copyright 2020 Intel Corporation.
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

#pragma once

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/dim_values.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

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
ShapeRef getShape(mlir::Value val);

//
// MemShape
//

using MemShape = details::DimValues<MemDim, int64_t, details::ShapeTag>;
using MemShapeRef = details::DimValuesRef<MemDim, int64_t, details::ShapeTag>;

MemShape getMemIndexND(int64_t memIndex1D, MemShapeRef memShape);
int64_t getMemIndex1D(MemShapeRef memIndexND, MemShapeRef memShape);

}  // namespace vpux

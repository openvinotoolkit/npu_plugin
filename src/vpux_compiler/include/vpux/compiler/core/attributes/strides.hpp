//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dim_values.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

#include "vpux/utils/core/mem_size.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

namespace vpux {

//
// StridesImpl
//

namespace details {

bool isDynamicDimValues(ArrayRef<Bit> strides);

template <class Base>
class StridesTag : public Base {
public:
    using Base::Base;

public:
    bool isDynamic() const {
        return details::isDynamicDimValues(this->raw());
    }

    bool isStatic() const {
        return !isDynamic();
    }
};

}  // namespace details

//
// Strides
//

using Strides = details::DimValues<Dim, Bit, details::StridesTag>;
using StridesRef = details::DimValuesRef<Dim, Bit, details::StridesTag>;

Strides getStrides(mlir::Value val);

//
// MemStrides
//

using MemStrides = details::DimValues<MemDim, Bit, details::StridesTag>;
using MemStridesRef = details::DimValuesRef<MemDim, Bit, details::StridesTag>;

MemStrides getMemStrides(mlir::Value val);

}  // namespace vpux

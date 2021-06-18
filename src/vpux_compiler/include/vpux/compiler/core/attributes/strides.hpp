//
// Copyright Intel Corporation.
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

Strides getStrides(mlir::MemRefType type);
Strides getStrides(mlir::Value val);

//
// MemStrides
//

using MemStrides = details::DimValues<MemDim, Bit, details::StridesTag>;
using MemStridesRef = details::DimValuesRef<MemDim, Bit, details::StridesTag>;

}  // namespace vpux

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

#include "vpux/compiler/core/dim_values.hpp"
#include "vpux/compiler/core/shape.hpp"

#include <mlir/IR/BuiltinTypes.h>

namespace vpux {

//
// StridesImpl
//

namespace details {

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

using Strides = details::DimValues<Dim, int64_t, details::StridesTag>;
using StridesRef = details::DimValuesRef<Dim, int64_t, details::StridesTag>;

Strides getStrides(mlir::MemRefType type);

int64_t getTypeByteSize(mlir::MemRefType type);

//
// MemStrides
//

using MemStrides = details::DimValues<MemDim, int64_t, details::StridesTag>;
using MemStridesRef = details::DimValuesRef<MemDim, int64_t, details::StridesTag>;

}  // namespace vpux

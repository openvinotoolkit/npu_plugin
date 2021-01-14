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
// TypeSize
//

Bit getElemTypeSize(mlir::Type type);
Byte getTypeTotalSize(mlir::MemRefType type);

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

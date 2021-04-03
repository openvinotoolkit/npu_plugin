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

#include "vpux/compiler/core/attributes/strides.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <algorithm>
#include <numeric>

using namespace vpux;

//
// Strides utils
//

bool vpux::details::isDynamicDimValues(ArrayRef<Bit> strides) {
    return std::any_of(strides.begin(), strides.end(), [](Bit val) {
        return val.count() <= 0;
    });
}

//
// TypeSize
//

Bit vpux::getElemTypeSize(mlir::Type type) {
    if (const auto shaped = type.dyn_cast<mlir::ShapedType>()) {
        return getElemTypeSize(shaped.getElementType());
    }

    if (type.isIntOrFloat()) {
        return Bit(type.getIntOrFloatBitWidth());
    }

    if (const auto qType = type.dyn_cast<mlir::quant::QuantizedType>()) {
        return Bit(qType.getStorageTypeIntegralWidth());
    }

    VPUX_THROW("Can't get type size for '{0}'", type);
}

Byte vpux::getTypeTotalSize(mlir::MemRefType type) {
    if (type.getRank() == 0) {
        return getElemTypeSize(type);
    }

    const auto dimsOrder = DimsOrder::fromType(type);
    const auto shape = getShape(type);
    const auto strides = getStrides(type);
    const auto memShape = dimsOrder.toMemoryOrder(shape);
    const auto memStrides = dimsOrder.toMemoryOrder(strides);

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Size and strides mismatch : {0} vs {1}", memShape,
                      memStrides);

    return Byte(memStrides.front() * memShape.front());
}

//
// Strides
//

Strides vpux::getStrides(mlir::MemRefType type) {
    const auto maps = type.getAffineMaps();

    if (maps.size() == 1 && maps.front().isPermutation()) {
        const auto dimsOrder = DimsOrder::fromAffineMap(maps.front());
        const auto memStrides = StrideReqs::simple(type.getRank()).calcStrides(dimsOrder, type);
        return dimsOrder.toLogicalOrder(memStrides);
    }

    SmallVector<int64_t> elemStrides;
    int64_t offset = 0;
    VPUX_THROW_UNLESS(mlir::succeeded(mlir::getStridesAndOffset(type, elemStrides, offset)),
                      "Only strided/simple MemRef Types are supported, got '{0}'", type);
    VPUX_THROW_UNLESS(elemStrides.size() == checked_cast<size_t>(type.getRank()),
                      "Only strided/simple MemRef Types are supported, got '{0}'", type);
    VPUX_THROW_UNLESS(offset == 0, "Only strided/simple MemRef Types are supported, got '{0}'", type);

    const auto elemSize = getElemTypeSize(type);
    Strides strides(elemStrides.size());

    for (auto i : irange(strides.size())) {
        strides[Dim(i)] = elemSize * elemStrides[i];
    }

    return strides;
}

Strides vpux::getStrides(mlir::Value val) {
    const auto type = val.getType().dyn_cast_or_null<mlir::MemRefType>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non MemRefType '{1}'", val, val.getType());
    return getStrides(type);
}

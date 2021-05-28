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

Byte vpux::getTotalSize(mlir::Value val) {
    const auto type = val.getType().dyn_cast_or_null<mlir::MemRefType>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non MemRefType '{1}'", val, val.getType());
    return getTypeTotalSize(type);
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

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

using namespace vpux;

Strides vpux::getStrides(mlir::MemRefType type) {
    const auto elemByteSize = type.getElementTypeBitWidth() / CHAR_BIT;

    if (const auto dimsOrder = DimsOrder::fromType(type)) {
        const auto shape = getShape(type);
        const auto memShape = dimsOrder->toMemoryOrder(shape);

        const auto reqs = StrideReqs::simple();
        const auto memStrides = reqs.calcStrides(elemByteSize, memShape);

        return dimsOrder->toLogicalOrder(memStrides);
    }

    Strides strides;
    int64_t offset = 0;
    VPUX_THROW_UNLESS(mlir::succeeded(mlir::getStridesAndOffset(type, strides.raw(), offset)),
                      "Only strided/simple MemRef Types are supported");
    VPUX_THROW_UNLESS(!strides.empty(), "Only strided/simple MemRef Types are supported");
    VPUX_THROW_UNLESS(offset == 0, "Only strided/simple MemRef Types are supported");

    for (auto& val : strides) {
        val *= elemByteSize;
    }

    return strides;
}

int64_t vpux::getTypeByteSize(mlir::MemRefType type) {
    const auto elemByteSize = type.getElementTypeBitWidth() / CHAR_BIT;

    if (type.getRank() == 0) {
        return elemByteSize;
    }

    const auto dimsOrder = DimsOrder::fromType(type);
    VPUX_THROW_UNLESS(dimsOrder.hasValue(), "Only strided/simple MemRef Types are supported");

    const auto shape = getShape(type);
    const auto strides = getStrides(type);

    const auto memShape = dimsOrder->toMemoryOrder(shape);
    const auto memStrides = dimsOrder->toMemoryOrder(strides);

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Size and strides mismatch : {0} vs {1}", memShape,
                      memStrides);

    return memStrides.back() * memShape.back();
}

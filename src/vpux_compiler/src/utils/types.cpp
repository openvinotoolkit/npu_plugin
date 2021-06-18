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

#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/core/attributes/strides.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

using namespace vpux;

namespace {

void setAffineMaps(mlir::MemRefType::Builder& memRefBuilder, mlir::MLIRContext* ctx, DimsOrder order, ShapeRef shape) {
    // MLIR has canonicalization for default permutations such as affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    // in this case we will only have a strided map, which means the same as having no maps at all.
    // Let's skip strided map as it makes IR and test writing easier.
    if (order == DimsOrder::fromNumDims(shape.size())) {
        memRefBuilder.setAffineMaps({});
        return;
    }

    const auto affineMaps = order.toAffineMapsList(ctx, shape);
    memRefBuilder.setAffineMaps(affineMaps);
}

}  // namespace

//
// get<scalar>Type
//

mlir::IntegerType vpux::getInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4);
}

mlir::IntegerType vpux::getInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8);
}

mlir::IntegerType vpux::getInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16);
}

mlir::IntegerType vpux::getInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32);
}

mlir::IntegerType vpux::getInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64);
}

mlir::IntegerType vpux::getSInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getUInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
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
// MemRefType utilities
//

mlir::MemRefType vpux::changeShape(mlir::MemRefType origType, ShapeRef shape) {
    const auto origOrder = DimsOrder::fromType(origType);
    auto memRefBuilder = mlir::MemRefType::Builder(origType).setShape(shape.raw());

    setAffineMaps(memRefBuilder, origType.getContext(), origOrder, shape);
    return memRefBuilder;
}

mlir::MemRefType vpux::changeDimsOrder(mlir::MemRefType origType, DimsOrder order) {
    const auto shape = getShape(origType);
    auto memRefBuilder = mlir::MemRefType::Builder(origType);

    setAffineMaps(memRefBuilder, origType.getContext(), order, shape);
    return memRefBuilder;
}

mlir::MemRefType vpux::changeElemType(mlir::MemRefType origType, mlir::Type elemType) {
    const auto origOrder = DimsOrder::fromType(origType);
    const auto shape = getShape(origType);
    auto memRefBuilder = mlir::MemRefType::Builder(origType);

    memRefBuilder.setElementType(elemType);
    setAffineMaps(memRefBuilder, origType.getContext(), origOrder, shape);

    return memRefBuilder;
}

mlir::MemRefType vpux::changeMemSpace(mlir::MemRefType origType, mlir::Attribute memSpace) {
    const auto origOrder = DimsOrder::fromType(origType);
    const auto shape = getShape(origType);
    auto memRefBuilder = mlir::MemRefType::Builder(origType);

    memRefBuilder.setMemorySpace(memSpace);
    setAffineMaps(memRefBuilder, origType.getContext(), origOrder, shape);

    return memRefBuilder;
}

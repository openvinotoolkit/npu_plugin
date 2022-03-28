//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/attributes/structs.hpp"
#include "vpux/compiler/dialect/IERT/attributes/structs.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

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

mlir::IntegerType vpux::getBool8Type(mlir::MLIRContext* ctx) {
    // Signless 8-bit integer use for BOOL, to distinguish it from U8
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signless);
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

Byte vpux::getTotalSize(mlir::Value val) {
    const auto type = val.getType().dyn_cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return type.getTotalAllocSize();
}

Byte vpux::getCompactSize(mlir::Value val) {
    const auto type = val.getType().dyn_cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return type.getCompactAllocSize();
}

Optional<int32_t> vpux::getQuantizedAxis(int32_t axis, ShapeRef prevShape, ShapeRef newShape) {
    auto prevArray = to_small_vector(prevShape.toValues());
    auto newArray = to_small_vector(newShape.toValues());

    if (checked_cast<size_t>(axis) >= prevShape.size()) {
        return None;
    }

    const auto isEqualOne = [](int64_t val) {
        return val == 1;
    };

    auto firstPrevIter = std::find_if_not(prevArray.begin(), prevArray.end(), isEqualOne);
    auto firstNewIter = std::find_if_not(newArray.begin(), newArray.end(), isEqualOne);

    if (firstPrevIter == prevArray.end() || firstNewIter == newArray.end()) {
        return axis;
    }

    const auto gap = checked_cast<int32_t>(std::distance(newArray.begin(), firstNewIter) -
                                           std::distance(prevArray.begin(), firstPrevIter));
    const auto newArraySize = newArray.size();

    prevArray.erase(std::remove_if(prevArray.begin(), prevArray.end(), isEqualOne), prevArray.end());
    newArray.erase(std::remove_if(newArray.begin(), newArray.end(), isEqualOne), newArray.end());

    if (!std::equal(prevArray.begin(), prevArray.end(), newArray.begin(), newArray.end())) {
        return None;
    }

    if (axis < -gap || axis + gap >= checked_cast<int32_t>(newArraySize)) {
        return None;
    }

    return axis + gap;
}

//
// MemRefType utilities
//

mlir::MemRefType vpux::getMemRefType(ShapeRef shape, mlir::Type elemType, DimsOrder order, IndexedSymbolAttr memSpace) {
    VPUX_THROW_UNLESS(order.numDims() == shape.size(), "Shape '{0}' doesn't match order '{1}'", shape, order);

    mlir::MemRefType::Builder builder(shape.raw(), elemType);
    builder.setLayout(mlir::AffineMapAttr::get(order.toAffineMap(elemType.getContext())));
    builder.setMemorySpace(memSpace);
    return builder;
}

mlir::MemRefType vpux::getMemRefType(ShapeRef shape, mlir::Type elemType, DimsOrder order, StridesRef strides,
                                     IndexedSymbolAttr memSpace) {
    VPUX_THROW_UNLESS(order.numDims() == shape.size(), "Shape '{0}' doesn't match order '{1}'", shape, order);
    VPUX_THROW_UNLESS(shape.size() == strides.size(), "Strides '{0}' doesn't match shape '{1}'", strides, shape);

    const auto elemSize = getElemTypeSize(elemType);

    const auto memShape = order.toMemoryOrder(shape);
    const auto memStrides = order.toMemoryOrder(strides);
    const auto compactReqs = StrideReqs::compact(shape.size());
    if (compactReqs.checkStrides(memStrides, elemSize, memShape)) {
        // Do not store compact strides.
        return getMemRefType(shape, elemType, order, memSpace);
    }

    const auto elemStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                 return stride.count() / elemSize.count();
                                             }));

    auto* ctx = elemType.getContext();
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
    const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
    const auto layoutAttr = IERT::MemRefAttr::get(orderAttr, stridesAttr, ctx);

    mlir::MemRefType::Builder builder(shape.raw(), elemType);
    builder.setLayout(layoutAttr.cast<mlir::MemRefLayoutAttrInterface>());
    builder.setMemorySpace(memSpace);
    return builder;
}

//
// RankedTensorType utilities
//

mlir::RankedTensorType vpux::getTensorType(ShapeRef shape, mlir::Type elemType, DimsOrder order,
                                           IndexedSymbolAttr memSpace, bool sparse) {
    VPUX_THROW_UNLESS(order.numDims() == shape.size(), "DimsOrder '{0}' doesn't match to shape '{1}'", order, shape);

    const auto tensorDesc = IE::getTensorAttr(elemType.getContext(), order, memSpace, sparse);
    const auto newType = mlir::RankedTensorType::get(shape.raw(), elemType, tensorDesc);

    const auto loc = mlir::UnknownLoc::get(elemType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::RankedTensorType vpux::changeSparse(mlir::RankedTensorType origType, bool sparse) {
    const auto ndType = origType.cast<vpux::NDTypeInterface>();
    return getTensorType(ndType.getShape(), ndType.getElementType(), ndType.getDimsOrder(), ndType.getMemSpace(),
                         sparse);
}

mlir::RankedTensorType vpux::getDilatedType(mlir::RankedTensorType tensor, vpux::ShapeRef dilations) {
    const auto type = tensor.cast<vpux::NDTypeInterface>();
    const auto targetRank = 4;
    VPUX_THROW_UNLESS(type.getRank() == targetRank, "Got invalid tensor rank '{0}'", targetRank);

    const auto origShape = type.getShape();

    const auto OC = origShape[vpux::Dims4D::Filter::OC];
    const auto IC = origShape[vpux::Dims4D::Filter::IC];
    const auto KY = origShape[vpux::Dims4D::Filter::KY];
    const auto KX = origShape[vpux::Dims4D::Filter::KX];

    // Calculate dilated kernel shape
    const auto dKY = KY + (KY - 1) * (dilations[Dim(0)] - 1);
    const auto dKX = KX + (KX - 1) * (dilations[Dim(1)] - 1);

    const auto dilatedShape = Shape({OC, IC, dKY, dKX});

    const auto newType = vpux::getTensorType(dilatedShape, type.getElementType(), type.getDimsOrder(),
                                             type.getMemSpace(), IE::isSparse(tensor));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

//
// ShapedType utilities
//

mlir::ShapedType vpux::changeSparse(mlir::ShapedType origType, bool sparse) {
    return llvm::TypeSwitch<mlir::ShapedType, mlir::ShapedType>(origType)
            .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType tensor) {
                return changeSparse(tensor, sparse);
            })
            .Default([](mlir::ShapedType type) -> mlir::ShapedType {
                VPUX_THROW("Unsupported ShapedType '{0}'", type);
            });
}

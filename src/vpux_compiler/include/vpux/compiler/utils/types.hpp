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

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/indexed_symbol_attr.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

#include "vpux/utils/core/mem_size.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

namespace vpux {

template <typename Enum, typename OutT = mlir::RankedTensorType>
using ranked_tensor_type_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

template <typename Enum, typename OutT = mlir::MemRefType>
using memref_type_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

//
// get<scalar>Type
//

mlir::IntegerType getInt4Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt8Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt16Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt32Type(mlir::MLIRContext* ctx);
mlir::IntegerType getInt64Type(mlir::MLIRContext* ctx);

mlir::IntegerType getSInt4Type(mlir::MLIRContext* ctx);
mlir::IntegerType getSInt8Type(mlir::MLIRContext* ctx);
mlir::IntegerType getSInt16Type(mlir::MLIRContext* ctx);
mlir::IntegerType getSInt32Type(mlir::MLIRContext* ctx);
mlir::IntegerType getSInt64Type(mlir::MLIRContext* ctx);

mlir::IntegerType getUInt4Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt8Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt16Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt32Type(mlir::MLIRContext* ctx);
mlir::IntegerType getUInt64Type(mlir::MLIRContext* ctx);
mlir::IntegerType getBoolType(mlir::MLIRContext* ctx);

//
// TypeSize
//

Bit getElemTypeSize(mlir::Type type);

// Calculates tensor size based on stride information
// Example:
// #map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 48 + d2 * 3 + d3)>
// memref<2x3x8x4xf32, #NHWC, #map0>
// result: 768 * 2 * 4 = 6144, where
// 768 - stride, to get the next element by N dim
// 2 - size of N dim, 4 - element size
Byte getTotalSize(mlir::ShapedType type);
Byte getTotalSize(mlir::Value val);

// Calculates tensor size ignoring stride information
// Example:
// #map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 48 + d2 * 3 + d3)>
// memref<2x3x8x4xui8, #NHWC, #map0>
// result: 2 * 3 * 8 * 4 = 192
Byte getCompactSize(mlir::ShapedType type);
Byte getCompactSize(mlir::Value val);

// compute axis permutation
Optional<int32_t> getQuantizedAxis(int32_t axis, ShapeRef prevShape, ShapeRef newShape);

//
// MemRefType utilities
//

mlir::MemRefType getMemRefType(ShapeRef shape, mlir::Type elemType, DimsOrder order, IndexedSymbolAttr memSpace);
template <typename Enum>
memref_type_if<Enum> getMemRefType(ShapeRef shape, mlir::Type elemType, DimsOrder order, Enum kind) {
    return getMemRefType(shape, elemType, order, IndexedSymbolAttr::get(elemType.getContext(), stringifyEnum(kind)));
}

mlir::MemRefType getMemRefType(ShapeRef shape, mlir::Type elemType, DimsOrder order, StridesRef strides,
                               IndexedSymbolAttr memSpace);
template <typename Enum>
memref_type_if<Enum> getMemRefType(ShapeRef shape, mlir::Type elemType, DimsOrder order, StridesRef strides,
                                   Enum kind) {
    return getMemRefType(shape, elemType, order, strides,
                         IndexedSymbolAttr::get(elemType.getContext(), stringifyEnum(kind)));
}

mlir::MemRefType changeElemType(mlir::MemRefType origType, mlir::Type elemType);
mlir::MemRefType changeShape(mlir::MemRefType origType, ShapeRef shape);
mlir::MemRefType changeDimsOrder(mlir::MemRefType origType, DimsOrder order);
mlir::MemRefType changeMemSpace(mlir::MemRefType origType, IndexedSymbolAttr memSpace);

template <typename Enum>
memref_type_if<Enum> changeMemSpace(mlir::MemRefType origType, Enum kind) {
    return changeMemSpace(origType, IndexedSymbolAttr::get(origType.getContext(), stringifyEnum(kind)));
}

mlir::MemRefType getDenseTileType(mlir::MemRefType origType, ShapeRef tileOffsets, ShapeRef tileShape);
mlir::MemRefType getViewTileType(mlir::MemRefType origType, ShapeRef tileOffsets, ShapeRef tileShape,
                                 ShapeRef tileElemStrides = {});
mlir::MemRefType getPaddedType(mlir::MemRefType origType, ShapeRef padBefore, ShapeRef padAfter);

mlir::MemRefType eraseTiledInfo(mlir::MemRefType origType);

IndexedSymbolAttr getMemorySpace(mlir::MemRefType type);

//
// RankedTensorType utilities
//

mlir::RankedTensorType getTensorType(ShapeRef shape, mlir::Type elemType, DimsOrder order, IndexedSymbolAttr memSpace,
                                     bool sparse = false);

mlir::RankedTensorType changeElemType(mlir::RankedTensorType origType, mlir::Type elemType);
mlir::RankedTensorType changeShape(mlir::RankedTensorType origType, ShapeRef shape);
mlir::RankedTensorType changeDimsOrder(mlir::RankedTensorType origType, DimsOrder order);
mlir::RankedTensorType changeSparse(mlir::RankedTensorType origType, bool sparse);

mlir::RankedTensorType changeMemSpace(mlir::RankedTensorType origType, IndexedSymbolAttr memSpace);
template <typename Enum>
ranked_tensor_type_if<Enum> changeMemSpace(mlir::RankedTensorType origType, Enum kind) {
    return changeMemSpace(origType, IndexedSymbolAttr::get(origType.getContext(), stringifyEnum(kind)));
}

mlir::RankedTensorType getDenseTileType(mlir::RankedTensorType origType, ShapeRef tileOffsets, ShapeRef tileShape);
mlir::RankedTensorType getPaddedType(mlir::RankedTensorType origType, ShapeRef padBefore, ShapeRef padAfter);

//
// ShapedType utilities
//

mlir::ShapedType changeElemType(mlir::ShapedType origType, mlir::Type elemType);
mlir::ShapedType changeShape(mlir::ShapedType origType, ShapeRef shape);
mlir::ShapedType changeDimsOrder(mlir::ShapedType origType, DimsOrder order);
mlir::ShapedType changeMemSpace(mlir::ShapedType origType, IndexedSymbolAttr memSpace);
mlir::ShapedType changeSparse(mlir::ShapedType origType, bool sparse);

mlir::ShapedType getDenseTileType(mlir::ShapedType origType, ShapeRef tileOffsets, ShapeRef tileShape);
mlir::ShapedType getPaddedType(mlir::ShapedType origType, ShapeRef padBefore, ShapeRef padAfter);

}  // namespace vpux

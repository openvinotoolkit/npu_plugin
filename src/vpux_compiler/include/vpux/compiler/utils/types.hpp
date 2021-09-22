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
#include "vpux/compiler/core/attributes/shape.hpp"

#include "vpux/utils/core/mem_size.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

namespace vpux {

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

//
// TypeSize
//

Bit getElemTypeSize(mlir::Type type);
Byte getTypeTotalSize(mlir::MemRefType type);
Byte getTotalSize(mlir::Value val);

//
// MemRefType utilities
//

mlir::MemRefType changeElemType(mlir::MemRefType origType, mlir::Type elemType, bool preserveStrides = false);
mlir::MemRefType changeShape(mlir::MemRefType origType, ShapeRef shape, bool preserveStrides = false);
mlir::MemRefType changeDimsOrder(mlir::MemRefType origType, DimsOrder order);
mlir::MemRefType changeMemSpace(mlir::MemRefType origType, mlir::Attribute memSpace, bool preserveStrides = false);

mlir::MemRefType getDenseTileType(mlir::MemRefType origType, ShapeRef tileOffsets, ShapeRef tileShape);
mlir::MemRefType getViewTileType(mlir::MemRefType origType, ShapeRef tileOffsets, ShapeRef tileShape,
                                 ShapeRef tileElemStrides = {});
mlir::MemRefType getPaddedType(mlir::MemRefType origType, ShapeRef padBefore, ShapeRef padAfter);

mlir::MemRefType eraseTiledInfo(mlir::MemRefType origType);

//
// RankedTensorType utilities
//

mlir::RankedTensorType getTensorType(ArrayRef<int64_t> shape, mlir::Type elementType, DimsOrder order);

mlir::RankedTensorType changeElemType(mlir::RankedTensorType origType, mlir::Type elemType);
mlir::RankedTensorType changeShape(mlir::RankedTensorType origType, ShapeRef shape);
mlir::RankedTensorType changeDimsOrder(mlir::RankedTensorType origType, DimsOrder order);

mlir::RankedTensorType getDenseTileType(mlir::RankedTensorType origType, ShapeRef tileOffsets, ShapeRef tileShape);
mlir::RankedTensorType getPaddedType(mlir::RankedTensorType origType, ShapeRef padBefore, ShapeRef padAfter);

//
// ShapedType utilities
//

mlir::ShapedType changeElemType(mlir::ShapedType origType, mlir::Type elemType);
mlir::ShapedType changeShape(mlir::ShapedType origType, ShapeRef shape);
mlir::ShapedType changeDimsOrder(mlir::ShapedType origType, DimsOrder order);

mlir::ShapedType getDenseTileType(mlir::ShapedType origType, ShapeRef tileOffsets, ShapeRef tileShape);
mlir::ShapedType getPaddedType(mlir::ShapedType origType, ShapeRef padBefore, ShapeRef padAfter);

}  // namespace vpux

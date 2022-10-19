//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/indexed_symbol_attr.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {

//
// TypeComponents
//

struct TypeComponents {
    Optional<ShapeRef> shape = None;
    Optional<mlir::Type> elementType = None;
    Optional<DimsOrder> dimsOrder = None;
    Optional<IndexedSymbolAttr> memSpace = None;
    Optional<StridesRef> strides = None;

    TypeComponents& setShape(ShapeRef newShape);
    TypeComponents& setElementType(mlir::Type newElementType);
    TypeComponents& setDimsOrder(DimsOrder newDimsOrder);
    TypeComponents& setMemSpace(IndexedSymbolAttr newMemSpace);
    TypeComponents& setStrides(StridesRef newStrides);
};

}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/core/generated/type_interfaces.hpp.inc>

namespace vpux {

class TensorNDTypeInterface : public NDTypeInterface::FallbackModel<TensorNDTypeInterface> {
public:
    vpux::ShapeRef getShape(mlir::Type type) const;
    vpux::MemShape getMemShape(mlir::Type type) const;
    bool hasRank(mlir::Type type) const;
    int64_t getRank(mlir::Type type) const;
    int64_t getNumElements(mlir::Type type) const;
    mlir::Type getElementType(mlir::Type type) const;
    vpux::DimsOrder getDimsOrder(mlir::Type type) const;
    vpux::IndexedSymbolAttr getMemSpace(mlir::Type type) const;
    vpux::VPU::MemoryKind getMemoryKind(mlir::Type type) const;
    vpux::Strides getStrides(mlir::Type type) const;
    vpux::MemStrides getMemStrides(mlir::Type type) const;
    vpux::Bit getElemTypeSize(mlir::Type type) const;
    vpux::Byte getTotalAllocSize(mlir::Type type) const;
    vpux::Byte getCompactAllocSize(mlir::Type type) const;
    vpux::NDTypeInterface changeShape(mlir::Type type, vpux::ShapeRef shape) const;
    vpux::NDTypeInterface changeElemType(mlir::Type type, mlir::Type elemType) const;
    vpux::NDTypeInterface changeShapeElemType(mlir::Type type, vpux::ShapeRef shape, mlir::Type elemType) const;
    vpux::NDTypeInterface changeDimsOrder(mlir::Type type, vpux::DimsOrder order) const;
    vpux::NDTypeInterface changeMemSpace(mlir::Type type, vpux::IndexedSymbolAttr memSpace) const;
    vpux::NDTypeInterface changeStrides(mlir::Type type, vpux::StridesRef strides) const;
    vpux::NDTypeInterface changeTypeComponents(mlir::Type type, vpux::TypeComponents typeComponents) const;
    vpux::NDTypeInterface extractDenseTile(mlir::Type type, vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape) const;
    vpux::NDTypeInterface extractViewTile(mlir::Type type, vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape,
                                          vpux::ShapeRef tileElemStrides) const;
    vpux::NDTypeInterface eraseTiledInfo(mlir::Type type) const;
    vpux::NDTypeInterface pad(mlir::Type type, vpux::ShapeRef padBefore, vpux::ShapeRef padAfter) const;
};

class MemRefNDTypeInterface : public vpux::NDTypeInterface::FallbackModel<MemRefNDTypeInterface> {
public:
    vpux::ShapeRef getShape(mlir::Type type) const;
    vpux::MemShape getMemShape(mlir::Type type) const;
    bool hasRank(mlir::Type type) const;
    int64_t getRank(mlir::Type type) const;
    int64_t getNumElements(mlir::Type type) const;
    mlir::Type getElementType(mlir::Type type) const;
    vpux::DimsOrder getDimsOrder(mlir::Type type) const;
    vpux::IndexedSymbolAttr getMemSpace(mlir::Type type) const;
    vpux::VPU::MemoryKind getMemoryKind(mlir::Type type) const;
    vpux::Strides getStrides(mlir::Type type) const;
    vpux::MemStrides getMemStrides(mlir::Type type) const;
    vpux::Bit getElemTypeSize(mlir::Type type) const;
    vpux::Byte getTotalAllocSize(mlir::Type type) const;
    vpux::Byte getCompactAllocSize(mlir::Type type) const;
    vpux::NDTypeInterface changeShape(mlir::Type type, vpux::ShapeRef shape) const;
    vpux::NDTypeInterface changeElemType(mlir::Type type, mlir::Type elemType) const;
    vpux::NDTypeInterface changeShapeElemType(mlir::Type type, vpux::ShapeRef shape, mlir::Type elemType) const;
    vpux::NDTypeInterface changeDimsOrder(mlir::Type type, vpux::DimsOrder order) const;
    vpux::NDTypeInterface changeMemSpace(mlir::Type type, vpux::IndexedSymbolAttr memSpace) const;
    vpux::NDTypeInterface changeStrides(mlir::Type type, vpux::StridesRef strides) const;
    vpux::NDTypeInterface changeTypeComponents(mlir::Type type, vpux::TypeComponents typeComponents) const;
    vpux::NDTypeInterface extractDenseTile(mlir::Type type, vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape) const;
    vpux::NDTypeInterface extractViewTile(mlir::Type type, vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape,
                                          vpux::ShapeRef tileElemStrides) const;
    vpux::NDTypeInterface eraseTiledInfo(mlir::Type type) const;
    vpux::NDTypeInterface pad(mlir::Type type, vpux::ShapeRef padBefore, vpux::ShapeRef padAfter) const;
};

}  // namespace vpux

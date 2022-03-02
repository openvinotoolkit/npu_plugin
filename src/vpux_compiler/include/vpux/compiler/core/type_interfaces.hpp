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
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

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
    vpux::NDTypeInterface extractDenseTile(mlir::Type type, vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape) const;
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
    vpux::NDTypeInterface extractDenseTile(mlir::Type type, vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape) const;
    vpux::NDTypeInterface pad(mlir::Type type, vpux::ShapeRef padBefore, vpux::ShapeRef padAfter) const;
};

}  // namespace vpux

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/types.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// print/parse
//

void VPU::SparseTensorType::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<data=" << getData();
    if (const auto& sparsityMap = getSparsityMap()) {
        printer << ", sparsity_map=" << sparsityMap;
    }
    if (const auto& storageElementTable = getStorageElementTable()) {
        printer << ", storage_element_table=" << storageElementTable;
    }
    printer << ">";
}

mlir::Type VPU::SparseTensorType::parse(mlir::DialectAsmParser& parser) {
    if (parser.parseLess())
        return Type();
    mlir::TensorType data;
    mlir::TensorType sparsityMap;
    mlir::TensorType storageElementTable;

    if (parser.parseKeyword("data")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::TensorType>(data)) {
        return Type();
    }
    if (!parser.parseOptionalGreater()) {
        return get(data, sparsityMap, storageElementTable);
    }

    if (parser.parseComma()) {
        return Type();
    }
    if (parser.parseKeyword("sparsity_map")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::TensorType>(sparsityMap)) {
        return Type();
    }
    if (!parser.parseOptionalGreater()) {
        return get(data, sparsityMap, storageElementTable);
    }

    if (parser.parseComma()) {
        return Type();
    }
    if (parser.parseKeyword("storage_element_table")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::TensorType>(storageElementTable)) {
        return Type();
    }
    if (parser.parseGreater()) {
        return Type();
    }

    return get(data, sparsityMap, storageElementTable);
}

//
// NDTypeInterface
//

ShapeRef VPU::SparseTensorType::getShape() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getShape();
}

MemShape VPU::SparseTensorType::getMemShape() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getMemShape();
}

bool VPU::SparseTensorType::hasRank() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.hasRank();
}

int64_t VPU::SparseTensorType::getRank() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getRank();
}

int64_t VPU::SparseTensorType::getNumElements() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getNumElements();
}

mlir::Type VPU::SparseTensorType::getElementType() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getElementType();
}

DimsOrder VPU::SparseTensorType::getDimsOrder() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getDimsOrder();
}

IndexedSymbolAttr VPU::SparseTensorType::getMemSpace() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getMemSpace();
}

VPU::MemoryKind VPU::SparseTensorType::getMemoryKind() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getMemoryKind();
}

Strides VPU::SparseTensorType::getStrides() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getStrides();
}

MemStrides VPU::SparseTensorType::getMemStrides() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getMemStrides();
}

Bit VPU::SparseTensorType::getElemTypeSize() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getElemTypeSize();
}

Byte VPU::SparseTensorType::getTotalAllocSize() const {
    const auto data = getData().cast<NDTypeInterface>();
    auto totalSize = data.getTotalAllocSize();
    if (getSparsityMap() != nullptr) {
        const auto sparsityMap = getSparsityMap().cast<NDTypeInterface>();
        totalSize += sparsityMap.getTotalAllocSize();
    }
    if (getStorageElementTable() != nullptr) {
        const auto storageElementTable = getStorageElementTable().cast<NDTypeInterface>();
        totalSize += storageElementTable.getTotalAllocSize();
    }
    return totalSize;
}

Byte VPU::SparseTensorType::getCompactAllocSize() const {
    const auto data = getData().cast<NDTypeInterface>();
    auto compactSize = data.getCompactAllocSize();
    if (getSparsityMap() != nullptr) {
        const auto sparsityMap = getSparsityMap().cast<NDTypeInterface>();
        compactSize += sparsityMap.getCompactAllocSize();
    }
    if (getStorageElementTable() != nullptr) {
        const auto storageElementTable = getStorageElementTable().cast<NDTypeInterface>();
        compactSize += storageElementTable.getCompactAllocSize();
    }
    return compactSize;
}

NDTypeInterface VPU::SparseTensorType::changeShape(ShapeRef shape) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeShape(shape).cast<mlir::TensorType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.changeShape(shape).cast<mlir::TensorType>();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        auto seTableShape = Shape(ndStorageElementTable.getShape().raw());
        seTableShape[Dims4D::Act::H] = shape[Dims4D::Act::H];
        seTableShape[Dims4D::Act::W] = shape[Dims4D::Act::W];
        storageElementTable = ndStorageElementTable.changeShape(seTableShape).cast<mlir::TensorType>();
    }

    return VPU::SparseTensorType::get(data, sparsityMap, storageElementTable);
}

NDTypeInterface VPU::SparseTensorType::changeElemType(mlir::Type elemType) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeElemType(elemType).cast<mlir::TensorType>();
    return VPU::SparseTensorType::get(data, getSparsityMap(), getStorageElementTable());
}

NDTypeInterface VPU::SparseTensorType::changeShapeElemType(ShapeRef shape, mlir::Type elemType) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeShapeElemType(shape, elemType).cast<mlir::TensorType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.changeShape(shape).cast<mlir::TensorType>();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        auto seTableShape = Shape(ndStorageElementTable.getShape().raw());
        seTableShape[Dims4D::Act::H] = shape[Dims4D::Act::H];
        seTableShape[Dims4D::Act::W] = shape[Dims4D::Act::W];
        storageElementTable = ndStorageElementTable.changeShape(seTableShape).cast<mlir::TensorType>();
    }

    return VPU::SparseTensorType::get(data, sparsityMap, storageElementTable);
}

NDTypeInterface VPU::SparseTensorType::changeDimsOrder(DimsOrder order) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeDimsOrder(order).cast<mlir::TensorType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.changeDimsOrder(order).cast<mlir::TensorType>();
    }

    return VPU::SparseTensorType::get(data, sparsityMap, getStorageElementTable());
}

NDTypeInterface VPU::SparseTensorType::changeMemSpace(IndexedSymbolAttr memSpace) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeMemSpace(memSpace).cast<mlir::TensorType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.changeMemSpace(memSpace).cast<mlir::TensorType>();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        storageElementTable = ndStorageElementTable.changeMemSpace(memSpace).cast<mlir::TensorType>();
    }

    return VPU::SparseTensorType::get(data, sparsityMap, storageElementTable);
}

NDTypeInterface VPU::SparseTensorType::changeStrides(StridesRef /*strides*/) const {
    VPUX_THROW("Sparse tensors only support compact strides");
}

NDTypeInterface VPU::SparseTensorType::changeTypeComponents(TypeComponents /*typeComponents*/) const {
    VPUX_THROW("changeTypeComponents method is not implemented for SparseTensorType");
}

NDTypeInterface VPU::SparseTensorType::extractDenseTile(ShapeRef tileOffsets, ShapeRef tileShape) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.extractDenseTile(tileOffsets, tileShape).cast<mlir::TensorType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.extractDenseTile(tileOffsets, tileShape).cast<mlir::TensorType>();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        auto seTableTileOffsets = Shape(tileOffsets.size(), 0);
        seTableTileOffsets[Dims4D::Act::H] = tileOffsets[Dims4D::Act::H];
        seTableTileOffsets[Dims4D::Act::W] = tileOffsets[Dims4D::Act::W];
        auto seTableTileShape = Shape(ndStorageElementTable.getShape().raw());
        seTableTileShape[Dims4D::Act::H] = tileShape[Dims4D::Act::H];
        seTableTileShape[Dims4D::Act::W] = tileShape[Dims4D::Act::W];
        storageElementTable =
                ndStorageElementTable.extractDenseTile(seTableTileOffsets, seTableTileShape).cast<mlir::TensorType>();
    }

    return VPU::SparseTensorType::get(data, sparsityMap, storageElementTable);
}

NDTypeInterface VPU::SparseTensorType::extractViewTile(ShapeRef /*tileOffsets*/, ShapeRef /*tileShape*/,
                                                       ShapeRef /*tileElemStrides*/) const {
    VPUX_THROW("Sparse tensors only support compact strides");
}

NDTypeInterface VPU::SparseTensorType::eraseTiledInfo() const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.eraseTiledInfo().cast<mlir::TensorType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.eraseTiledInfo().cast<mlir::TensorType>();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        storageElementTable = ndStorageElementTable.eraseTiledInfo().cast<mlir::TensorType>();
    }

    return VPU::SparseTensorType::get(data, sparsityMap, storageElementTable);
}

NDTypeInterface VPU::SparseTensorType::pad(ShapeRef padBefore, ShapeRef padAfter) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.pad(padBefore, padAfter).cast<mlir::TensorType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.pad(padBefore, padAfter).cast<mlir::TensorType>();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        const Shape seTablePadBefore{0, 0, padBefore[Dims4D::Act::H], padBefore[Dims4D::Act::W]};
        const Shape seTablePadAfter{0, 0, padAfter[Dims4D::Act::H], padAfter[Dims4D::Act::W]};
        storageElementTable = ndStorageElementTable.pad(seTablePadBefore, seTablePadAfter).cast<mlir::TensorType>();
    }

    return VPU::SparseTensorType::get(data, sparsityMap, storageElementTable);
}

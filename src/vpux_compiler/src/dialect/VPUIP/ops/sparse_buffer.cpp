//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

//
// build
//

void VPUIP::GroupSparseBufferOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value data) {
    build(odsBuilder, odsState, data, nullptr, nullptr);
}

void VPUIP::GroupSparseBufferOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value data,
                                       mlir::Value sparsityMap) {
    build(odsBuilder, odsState, data, sparsityMap, nullptr);
}

//
// getViewSources
//

mlir::ValueRange VPUIP::GroupSparseBufferOp::getViewSources() {
    return getOperands();
}

//
// inferReturnTypes
//

mlir::LogicalResult VPUIP::GroupSparseBufferOp::inferReturnTypes(mlir::MLIRContext*, Optional<mlir::Location>,
                                                                 mlir::ValueRange operands,
                                                                 mlir::DictionaryAttr /*attrs*/,
                                                                 mlir::RegionRange /*ranges*/,
                                                                 SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto data = operands[0];
    const auto sparsityMap = operands.size() > 1 ? operands[1] : nullptr;
    const auto storageElementTable = operands.size() > 2 ? operands[2] : nullptr;

    const auto dataType = data.getType().cast<mlir::MemRefType>();
    const auto sparsityMapType = sparsityMap ? sparsityMap.getType().cast<mlir::MemRefType>() : nullptr;
    const auto storageElementTableType =
            storageElementTable ? storageElementTable.getType().cast<mlir::MemRefType>() : nullptr;

    inferredReturnTypes.push_back(VPUIP::SparseBufferType::get(dataType, sparsityMapType, storageElementTableType));

    return mlir::success();
}

//
// print/parse
//

void VPUIP::SparseBufferType::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<data=" << getData();
    if (const auto& sparsityMap = getSparsityMap()) {
        printer << ", sparsity_map=" << sparsityMap;
    }
    if (const auto& storageElementTable = getStorageElementTable()) {
        printer << ", storage_element_table=" << storageElementTable;
    }
    printer << ">";
}

mlir::Type VPUIP::SparseBufferType::parse(mlir::DialectAsmParser& parser) {
    if (parser.parseLess())
        return Type();
    mlir::MemRefType data;
    mlir::MemRefType sparsityMap;
    mlir::MemRefType storageElementTable;

    if (parser.parseKeyword("data")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::MemRefType>(data)) {
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
    if (parser.parseType<mlir::MemRefType>(sparsityMap)) {
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
    if (parser.parseType<mlir::MemRefType>(storageElementTable)) {
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

ShapeRef VPUIP::SparseBufferType::getShape() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getShape();
}

MemShape VPUIP::SparseBufferType::getMemShape() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getMemShape();
}

bool VPUIP::SparseBufferType::hasRank() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.hasRank();
}

int64_t VPUIP::SparseBufferType::getRank() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getRank();
}

int64_t VPUIP::SparseBufferType::getNumElements() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getNumElements();
}

mlir::Type VPUIP::SparseBufferType::getElementType() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getElementType();
}

DimsOrder VPUIP::SparseBufferType::getDimsOrder() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getDimsOrder();
}

IndexedSymbolAttr VPUIP::SparseBufferType::getMemSpace() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getMemSpace();
}

VPU::MemoryKind VPUIP::SparseBufferType::getMemoryKind() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getMemoryKind();
}

Strides VPUIP::SparseBufferType::getStrides() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getStrides();
}

MemStrides VPUIP::SparseBufferType::getMemStrides() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getMemStrides();
}

Bit VPUIP::SparseBufferType::getElemTypeSize() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getElemTypeSize();
}

Byte VPUIP::SparseBufferType::getTotalAllocSize() const {
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

Byte VPUIP::SparseBufferType::getCompactAllocSize() const {
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

NDTypeInterface VPUIP::SparseBufferType::changeShape(ShapeRef shape) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeShape(shape).cast<mlir::MemRefType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.changeShape(shape).cast<mlir::MemRefType>();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        auto seTableShape = Shape(ndStorageElementTable.getShape().raw());
        seTableShape[Dims4D::Act::H] = shape[Dims4D::Act::H];
        seTableShape[Dims4D::Act::W] = shape[Dims4D::Act::W];
        storageElementTable = ndStorageElementTable.changeShape(seTableShape).cast<mlir::MemRefType>();
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable);
}

NDTypeInterface VPUIP::SparseBufferType::changeElemType(mlir::Type elemType) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeElemType(elemType).cast<mlir::MemRefType>();
    return VPUIP::SparseBufferType::get(data, getSparsityMap(), getStorageElementTable());
}

NDTypeInterface VPUIP::SparseBufferType::changeShapeElemType(ShapeRef shape, mlir::Type elemType) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeShapeElemType(shape, elemType).cast<mlir::MemRefType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.changeShape(shape).cast<mlir::MemRefType>();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        auto seTableShape = Shape(ndStorageElementTable.getShape().raw());
        seTableShape[Dims4D::Act::H] = shape[Dims4D::Act::H];
        seTableShape[Dims4D::Act::W] = shape[Dims4D::Act::W];
        storageElementTable = ndStorageElementTable.changeShape(seTableShape).cast<mlir::MemRefType>();
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable);
}

NDTypeInterface VPUIP::SparseBufferType::changeDimsOrder(DimsOrder order) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeDimsOrder(order).cast<mlir::MemRefType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.changeDimsOrder(order).cast<mlir::MemRefType>();
    }

    // The order of the storage element table should not be changed since it is always 1xDxHxW
    const auto storageElementTable = getStorageElementTable();

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable);
}

NDTypeInterface VPUIP::SparseBufferType::changeMemSpace(IndexedSymbolAttr memSpace) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeMemSpace(memSpace).cast<mlir::MemRefType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.changeMemSpace(memSpace).cast<mlir::MemRefType>();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        storageElementTable = ndStorageElementTable.changeMemSpace(memSpace).cast<mlir::MemRefType>();
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable);
}

NDTypeInterface VPUIP::SparseBufferType::changeStrides(StridesRef strides) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeStrides(strides).cast<mlir::MemRefType>();
    return VPUIP::SparseBufferType::get(data, getSparsityMap(), getStorageElementTable());
}

NDTypeInterface VPUIP::SparseBufferType::changeTypeComponents(TypeComponents /*typeComponents*/) const {
    VPUX_THROW("changeTypeComponents method is not implemented for SparseBufferType");
}

NDTypeInterface VPUIP::SparseBufferType::extractDenseTile(ShapeRef tileOffsets, ShapeRef tileShape) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.extractDenseTile(tileOffsets, tileShape).cast<mlir::MemRefType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.extractDenseTile(tileOffsets, tileShape).cast<mlir::MemRefType>();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        auto seTableTileOffsets = Shape(tileOffsets.size(), 1);
        seTableTileOffsets[Dims4D::Act::H] = tileOffsets[Dims4D::Act::H];
        seTableTileOffsets[Dims4D::Act::W] = tileOffsets[Dims4D::Act::W];
        auto seTableTileShape = Shape(ndStorageElementTable.getShape().raw());
        seTableTileShape[Dims4D::Act::H] = tileShape[Dims4D::Act::H];
        seTableTileShape[Dims4D::Act::W] = tileShape[Dims4D::Act::W];
        storageElementTable =
                ndStorageElementTable.extractDenseTile(seTableTileOffsets, seTableTileShape).cast<mlir::MemRefType>();
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable);
}

NDTypeInterface VPUIP::SparseBufferType::extractViewTile(ShapeRef tileOffsets, ShapeRef tileShape,
                                                         ShapeRef tileElemStrides) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.extractViewTile(tileOffsets, tileShape, tileElemStrides).cast<mlir::MemRefType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.extractDenseTile(tileOffsets, tileShape).cast<mlir::MemRefType>();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        auto seTableTileOffsets = Shape(tileOffsets.size(), 1);
        seTableTileOffsets[Dims4D::Act::H] = tileOffsets[Dims4D::Act::H];
        seTableTileOffsets[Dims4D::Act::W] = tileOffsets[Dims4D::Act::W];
        auto seTableTileShape = Shape(ndStorageElementTable.getShape().raw());
        seTableTileShape[Dims4D::Act::H] = tileShape[Dims4D::Act::H];
        seTableTileShape[Dims4D::Act::W] = tileShape[Dims4D::Act::W];
        storageElementTable =
                ndStorageElementTable.extractDenseTile(seTableTileOffsets, seTableTileShape).cast<mlir::MemRefType>();
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable);
}

NDTypeInterface VPUIP::SparseBufferType::eraseTiledInfo() const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.eraseTiledInfo().cast<mlir::MemRefType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.eraseTiledInfo().cast<mlir::MemRefType>();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        storageElementTable = ndStorageElementTable.eraseTiledInfo().cast<mlir::MemRefType>();
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable);
}

NDTypeInterface VPUIP::SparseBufferType::pad(ShapeRef padBefore, ShapeRef padAfter) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.pad(padBefore, padAfter).cast<mlir::MemRefType>();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.pad(padBefore, padAfter).cast<mlir::MemRefType>();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        const Shape seTablePadBefore{0, 0, padBefore[Dims4D::Act::H], padBefore[Dims4D::Act::W]};
        const Shape seTablePadAfter{0, 0, padAfter[Dims4D::Act::H], padAfter[Dims4D::Act::W]};
        storageElementTable = ndStorageElementTable.pad(seTablePadBefore, seTablePadAfter).cast<mlir::MemRefType>();
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable);
}

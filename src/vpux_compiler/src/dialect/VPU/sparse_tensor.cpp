//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <numeric>

using namespace vpux;

//
// print/parse
//

void VPU::SparseTensorType::print(mlir::AsmPrinter& printer) const {
    printer << "<data=" << getData();
    if (const auto& sparsityMap = getSparsityMap()) {
        printer << ", sparsity_map=" << sparsityMap;
    }
    if (const auto& storageElementTable = getStorageElementTable()) {
        printer << ", storage_element_table=" << storageElementTable;
    }
    if (getIsWeights() != nullptr) {
        printer << ", is_weights";
    }
    printer << ">";
}

mlir::Type VPU::SparseTensorType::parse(mlir::AsmParser& parser) {
    if (parser.parseLess())
        return Type();
    mlir::Type data;
    mlir::Type sparsityMap;
    mlir::Type storageElementTable;

    if (parser.parseKeyword("data")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::Type>(data)) {
        return Type();
    }
    if (!parser.parseOptionalGreater()) {
        return get(data, sparsityMap, storageElementTable);
    }

    if (parser.parseComma()) {
        return Type();
    }
    if (!parser.parseOptionalKeyword("is_weights")) {
        if (parser.parseGreater()) {
            return Type();
        }
        return get(data, sparsityMap, storageElementTable, mlir::UnitAttr::get(parser.getContext()));
    }
    if (parser.parseKeyword("sparsity_map")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::Type>(sparsityMap)) {
        return Type();
    }
    if (!parser.parseOptionalGreater()) {
        return get(data, sparsityMap, storageElementTable);
    }

    if (parser.parseComma()) {
        return Type();
    }
    if (!parser.parseOptionalKeyword("is_weights")) {
        if (parser.parseGreater()) {
            return Type();
        }
        return get(data, sparsityMap, storageElementTable, mlir::UnitAttr::get(parser.getContext()));
    }
    if (parser.parseKeyword("storage_element_table")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::Type>(storageElementTable)) {
        return Type();
    }
    if (!parser.parseOptionalComma()) {
        if (parser.parseKeyword("is_weights")) {
            return Type();
        }
        if (parser.parseGreater()) {
            return Type();
        }
        return get(data, sparsityMap, storageElementTable, mlir::UnitAttr::get(parser.getContext()));
    }
    if (parser.parseGreater()) {
        return Type();
    }

    return get(data, sparsityMap, storageElementTable);
}

//
// verify
//

mlir::LogicalResult VPU::SparseTensorType::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                                  mlir::Type data, mlir::Type sparsityMap, mlir::Type seTable,
                                                  mlir::UnitAttr /*isWeights*/) {
    if (!data.isa<mlir::RankedTensorType, VPU::DistributedTensorType>()) {
        return printTo(emitError(), "Data type is not a ranked or distributed tensor. Got {0}", data);
    }
    if (sparsityMap != nullptr && !sparsityMap.isa<mlir::RankedTensorType, VPU::DistributedTensorType>()) {
        return printTo(emitError(), "Sparsity map type is not a ranked or distributed tensor. Got {0}", sparsityMap);
    }
    if (seTable != nullptr && !seTable.isa<mlir::RankedTensorType, VPU::DistributedTensorType>()) {
        return printTo(emitError(), "Storage element table type is not a ranked or distributed tensor. Got {0}",
                       seTable);
    }

    if (data.isa<VPU::DistributedTensorType>()) {
        if (sparsityMap != nullptr && !sparsityMap.isa<VPU::DistributedTensorType>()) {
            return printTo(emitError(), "Sparsity map of type {0} is not a distributed tensor while data is",
                           sparsityMap);
        }
        if (seTable != nullptr && !seTable.isa<VPU::DistributedTensorType>()) {
            return printTo(emitError(), "Storage element table of type {0} is not a distributed tensor while data is",
                           seTable);
        }
    }

    return mlir::success();
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
    return changeShapeElemType(shape, getElementType());
}

NDTypeInterface VPU::SparseTensorType::changeElemType(mlir::Type elemType) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeElemType(elemType);
    return VPU::SparseTensorType::get(data, getSparsityMap(), getStorageElementTable(), getIsWeights());
}

NDTypeInterface VPU::SparseTensorType::changeShapeElemType(ShapeRef shape, mlir::Type elemType) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeShapeElemType(shape, elemType);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        if (getIsWeights() != nullptr) {
            auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(data.getShape());
            sparsityMap = ndSparsityMap.changeShape(newSMShape);
        } else {
            sparsityMap = ndSparsityMap.changeShape(shape);
        }
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        auto seTableShape = Shape(ndStorageElementTable.getShape().raw());
        seTableShape[Dims4D::Act::H] = shape[Dims4D::Act::H];
        seTableShape[Dims4D::Act::W] = shape[Dims4D::Act::W];
        storageElementTable = ndStorageElementTable.changeShape(seTableShape);
    }

    return VPU::SparseTensorType::get(data, sparsityMap, storageElementTable, getIsWeights());
}

NDTypeInterface VPU::SparseTensorType::changeDimsOrder(DimsOrder order) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeDimsOrder(order);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        if (getIsWeights() == nullptr) {
            sparsityMap = ndSparsityMap.changeDimsOrder(order);
        }
    }

    return VPU::SparseTensorType::get(data, sparsityMap, getStorageElementTable(), getIsWeights());
}

NDTypeInterface VPU::SparseTensorType::changeMemSpace(IndexedSymbolAttr memSpace) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeMemSpace(memSpace);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.changeMemSpace(memSpace);
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        storageElementTable = ndStorageElementTable.changeMemSpace(memSpace);
    }

    return VPU::SparseTensorType::get(data, sparsityMap, storageElementTable, getIsWeights());
}

NDTypeInterface VPU::SparseTensorType::changeStrides(StridesRef /*strides*/) const {
    VPUX_THROW("Sparse tensors only support compact strides");
}

NDTypeInterface VPU::SparseTensorType::changeTypeComponents(TypeComponents /*typeComponents*/) const {
    VPUX_THROW("changeTypeComponents method is not implemented for SparseTensorType");
}

NDTypeInterface VPU::SparseTensorType::extractDenseTile(ShapeRef tileOffsets, ShapeRef tileShape) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.extractDenseTile(tileOffsets, tileShape);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        if (getIsWeights() != nullptr) {
            auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(data.getShape());
            sparsityMap = ndSparsityMap.changeShape(newSMShape);
        } else {
            sparsityMap = ndSparsityMap.extractDenseTile(tileOffsets, tileShape);
        }
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
        storageElementTable = ndStorageElementTable.extractDenseTile(seTableTileOffsets, seTableTileShape);
    }

    return VPU::SparseTensorType::get(data, sparsityMap, storageElementTable, getIsWeights());
}

NDTypeInterface VPU::SparseTensorType::extractViewTile(ShapeRef /*tileOffsets*/, ShapeRef /*tileShape*/,
                                                       ShapeRef /*tileElemStrides*/) const {
    VPUX_THROW("Sparse tensors only support compact strides");
}

NDTypeInterface VPU::SparseTensorType::eraseTiledInfo() const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.eraseTiledInfo();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        sparsityMap = ndSparsityMap.eraseTiledInfo();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        storageElementTable = ndStorageElementTable.eraseTiledInfo();
    }

    return VPU::SparseTensorType::get(data, sparsityMap, storageElementTable, getIsWeights());
}

NDTypeInterface VPU::SparseTensorType::pad(ShapeRef padBefore, ShapeRef padAfter) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.pad(padBefore, padAfter);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<NDTypeInterface>();
        if (getIsWeights() != nullptr) {
            auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(data.getShape());
            sparsityMap = ndSparsityMap.changeShape(newSMShape);
        } else {
            sparsityMap = ndSparsityMap.pad(padBefore, padAfter);
        }
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<NDTypeInterface>();
        const Shape seTablePadBefore{0, 0, padBefore[Dims4D::Act::H], padBefore[Dims4D::Act::W]};
        const Shape seTablePadAfter{0, 0, padAfter[Dims4D::Act::H], padAfter[Dims4D::Act::W]};
        storageElementTable = ndStorageElementTable.pad(seTablePadBefore, seTablePadAfter);
    }

    return VPU::SparseTensorType::get(data, sparsityMap, storageElementTable, getIsWeights());
}

//
// DistributedTypeInterface
//

bool VPU::SparseTensorType::containsDistributedTypes() const {
    // If the data is a distributed type, the metadata will be as well
    return getData().isa<VPU::DistributedTensorType>();
}

SmallVector<mlir::Type> VPU::SparseTensorType::getDistributedTypes() const {
    SmallVector<mlir::Type> distributedTypes;
    if (getData().isa<VPU::DistributedTensorType>()) {
        distributedTypes.push_back(getData());
    }
    if (getSparsityMap() != nullptr && getSparsityMap().isa<VPU::DistributedTensorType>()) {
        distributedTypes.push_back(getSparsityMap());
    }
    if (getStorageElementTable() != nullptr && getStorageElementTable().isa<VPU::DistributedTensorType>()) {
        distributedTypes.push_back(getStorageElementTable());
    }
    return distributedTypes;
}

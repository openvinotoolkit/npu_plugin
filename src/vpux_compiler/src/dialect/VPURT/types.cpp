//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/dialect/VPURT/types.hpp"

#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPURT/generated/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// VPURTDialect::registerTypes
//

void vpux::VPURT::VPURTDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPURT/generated/types.cpp.inc>
            >();
}

//
// Dialect hooks
//

mlir::Type vpux::VPURT::VPURTDialect::parseType(mlir::DialectAsmParser& parser) const {
    StringRef mnemonic;
    if (mlir::failed(parser.parseKeyword(&mnemonic))) {
        printTo(parser.emitError(parser.getCurrentLocation()), "Failed to get VPURT Type mnemonic");
        return nullptr;
    }

    mlir::Type type;
    if (!generatedTypeParser(parser, mnemonic, type).hasValue()) {
        printTo(parser.emitError(parser.getCurrentLocation()), "Unknown VPURT Type '{0}'", mnemonic);
    }

    return type;
}

void vpux::VPURT::VPURTDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& os) const {
    VPUX_THROW_UNLESS(mlir::succeeded(generatedTypePrinter(type, os)), "Got unsupported Type : {0}", type);
}

//
// SparseBufferType
//

mlir::Value vpux::VPURT::SparseBufferType::getData(mlir::Value val) {
    if (auto sparseBuffer = val.getDefiningOp<vpux::VPURT::DeclareSparseBufferOp>()) {
        return sparseBuffer.data();
    }
    return val;
}

mlir::Value vpux::VPURT::SparseBufferType::getSparsityMap(mlir::Value val) {
    if (auto sparseBuffer = val.getDefiningOp<vpux::VPURT::DeclareSparseBufferOp>()) {
        return sparseBuffer.sparsityMap();
    }
    return nullptr;
}

mlir::Value vpux::VPURT::SparseBufferType::getStorageElementTable(mlir::Value val) {
    if (auto sparseBuffer = val.getDefiningOp<vpux::VPURT::DeclareSparseBufferOp>()) {
        return sparseBuffer.storageElementTable();
    }
    return nullptr;
}

mlir::Type vpux::VPURT::SparseBufferType::getDataType(mlir::Value val) {
    return getData(val).getType();
}

void vpux::VPURT::SparseBufferType::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<data=" << getData();
    if (const auto& sparsityMap = getSparsity_map()) {
        printer << ", sparsity_map=" << sparsityMap;
    }
    if (const auto& storageElementTable = getStorage_element_table()) {
        printer << ", storage_element_table=" << storageElementTable;
    }
    printer << ">";
}

mlir::Type vpux::VPURT::SparseBufferType::parse(mlir::DialectAsmParser& parser) {
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

vpux::ShapeRef vpux::VPURT::SparseBufferType::getShape() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    return data.getShape();
}

vpux::MemShape vpux::VPURT::SparseBufferType::getMemShape() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    return data.getMemShape();
}

bool vpux::VPURT::SparseBufferType::hasRank() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    return data.hasRank();
}

int64_t vpux::VPURT::SparseBufferType::getRank() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    return data.getRank();
}

int64_t vpux::VPURT::SparseBufferType::getNumElements() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    return data.getNumElements();
}

mlir::Type vpux::VPURT::SparseBufferType::getElementType() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    return data.getElementType();
}

vpux::DimsOrder vpux::VPURT::SparseBufferType::getDimsOrder() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    return data.getDimsOrder();
}

vpux::IndexedSymbolAttr vpux::VPURT::SparseBufferType::getMemSpace() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    return data.getMemSpace();
}

vpux::VPU::MemoryKind vpux::VPURT::SparseBufferType::getMemoryKind() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    return data.getMemoryKind();
}

vpux::Strides vpux::VPURT::SparseBufferType::getStrides() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    return data.getStrides();
}

vpux::MemStrides vpux::VPURT::SparseBufferType::getMemStrides() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    return data.getMemStrides();
}

vpux::Bit vpux::VPURT::SparseBufferType::getElemTypeSize() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    return data.getElemTypeSize();
}

vpux::Byte vpux::VPURT::SparseBufferType::getTotalAllocSize() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    auto totalSize = data.getTotalAllocSize();
    if (getSparsity_map() != nullptr) {
        const auto sparsityMap = getSparsity_map().cast<vpux::NDTypeInterface>();
        totalSize += sparsityMap.getTotalAllocSize();
    }
    if (getStorage_element_table() != nullptr) {
        const auto storageElementTable = getStorage_element_table().cast<vpux::NDTypeInterface>();
        totalSize += storageElementTable.getTotalAllocSize();
    }
    return totalSize;
}

vpux::Byte vpux::VPURT::SparseBufferType::getCompactAllocSize() const {
    const auto data = getData().cast<vpux::NDTypeInterface>();
    auto compactSize = data.getCompactAllocSize();
    if (getSparsity_map() != nullptr) {
        const auto sparsityMap = getSparsity_map().cast<vpux::NDTypeInterface>();
        compactSize += sparsityMap.getCompactAllocSize();
    }
    if (getStorage_element_table() != nullptr) {
        const auto storageElementTable = getStorage_element_table().cast<vpux::NDTypeInterface>();
        compactSize += storageElementTable.getCompactAllocSize();
    }
    return compactSize;
}

vpux::NDTypeInterface vpux::VPURT::SparseBufferType::changeShape(vpux::ShapeRef shape) const {
    VPUX_THROW_UNLESS(getStorage_element_table() == nullptr, "Storage element table is not yet supported");

    const auto ndData = getData().cast<vpux::NDTypeInterface>();
    const auto data = ndData.changeShape(shape).cast<mlir::MemRefType>();

    auto sparsityMap = getSparsity_map();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<vpux::NDTypeInterface>();
        sparsityMap = ndSparsityMap.changeShape(shape).cast<mlir::MemRefType>();
    }

    return VPURT::SparseBufferType::get(data, sparsityMap);
}

vpux::NDTypeInterface vpux::VPURT::SparseBufferType::changeElemType(mlir::Type elemType) const {
    const auto ndData = getData().cast<vpux::NDTypeInterface>();
    const auto data = ndData.changeElemType(elemType).cast<mlir::MemRefType>();
    return VPURT::SparseBufferType::get(data, getSparsity_map(), getStorage_element_table());
}

vpux::NDTypeInterface vpux::VPURT::SparseBufferType::changeShapeElemType(vpux::ShapeRef shape,
                                                                         mlir::Type elemType) const {
    VPUX_THROW_UNLESS(getStorage_element_table() == nullptr, "Storage element table is not yet supported");

    const auto ndData = getData().cast<vpux::NDTypeInterface>();
    const auto data = ndData.changeShapeElemType(shape, elemType).cast<mlir::MemRefType>();

    auto sparsityMap = getSparsity_map();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<vpux::NDTypeInterface>();
        sparsityMap = ndSparsityMap.changeShape(shape).cast<mlir::MemRefType>();
    }

    return VPURT::SparseBufferType::get(data, sparsityMap);
}

vpux::NDTypeInterface vpux::VPURT::SparseBufferType::changeDimsOrder(vpux::DimsOrder order) const {
    const auto ndData = getData().cast<vpux::NDTypeInterface>();
    const auto data = ndData.changeDimsOrder(order).cast<mlir::MemRefType>();

    auto sparsityMap = getSparsity_map();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<vpux::NDTypeInterface>();
        sparsityMap = ndSparsityMap.changeDimsOrder(order).cast<mlir::MemRefType>();
    }

    auto storageElementTable = getStorage_element_table();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<vpux::NDTypeInterface>();
        storageElementTable = ndStorageElementTable.changeDimsOrder(order).cast<mlir::MemRefType>();
    }

    return VPURT::SparseBufferType::get(data, sparsityMap, storageElementTable);
}

vpux::NDTypeInterface vpux::VPURT::SparseBufferType::changeMemSpace(vpux::IndexedSymbolAttr memSpace) const {
    const auto ndData = getData().cast<vpux::NDTypeInterface>();
    const auto data = ndData.changeMemSpace(memSpace).cast<mlir::MemRefType>();

    auto sparsityMap = getSparsity_map();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = sparsityMap.cast<vpux::NDTypeInterface>();
        sparsityMap = ndSparsityMap.changeMemSpace(memSpace).cast<mlir::MemRefType>();
    }

    auto storageElementTable = getStorage_element_table();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = storageElementTable.cast<vpux::NDTypeInterface>();
        storageElementTable = ndStorageElementTable.changeMemSpace(memSpace).cast<mlir::MemRefType>();
    }

    return VPURT::SparseBufferType::get(data, sparsityMap, storageElementTable);
}

vpux::NDTypeInterface vpux::VPURT::SparseBufferType::changeStrides(StridesRef /*strides*/) const {
    VPUX_THROW("changeStrides method is not implemented for SparseBufferType");
}

vpux::NDTypeInterface vpux::VPURT::SparseBufferType::extractDenseTile(vpux::ShapeRef /*tileOffsets*/,
                                                                      vpux::ShapeRef /*tileShape*/) const {
    VPUX_THROW("extractDenseTile method is not implemented for SparseBufferType");
}

vpux::NDTypeInterface vpux::VPURT::SparseBufferType::extractViewTile(vpux::ShapeRef /*tileOffsets*/,
                                                                     vpux::ShapeRef /*tileShape*/,
                                                                     vpux::ShapeRef /*tileElemStrides*/) const {
    VPUX_THROW("extractViewTile method is not implemented for DistributedBufferType");
}

vpux::NDTypeInterface vpux::VPURT::SparseBufferType::eraseTiledInfo() const {
    VPUX_THROW("eraseTiledInfo method is not implemented for DistributedBufferType");
}

vpux::NDTypeInterface vpux::VPURT::SparseBufferType::pad(vpux::ShapeRef /*padBefore*/,
                                                         vpux::ShapeRef /*padAfter*/) const {
    VPUX_THROW("pad method is not implemented for SparseBufferType");
}

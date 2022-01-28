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

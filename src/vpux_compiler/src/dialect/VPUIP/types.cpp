//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/types.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPUIP/generated/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// VPUIPDialect::registerTypes
//

void vpux::VPUIP::VPUIPDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPUIP/generated/types.cpp.inc>
            >();
}

//
// Dialect hooks
//

mlir::Type vpux::VPUIP::VPUIPDialect::parseType(mlir::DialectAsmParser& parser) const {
    StringRef mnemonic;
    if (mlir::failed(parser.parseKeyword(&mnemonic))) {
        printTo(parser.emitError(parser.getCurrentLocation()), "Failed to get VPUIP Type mnemonic");
        return nullptr;
    }

    mlir::Type type;
    if (!generatedTypeParser(parser, mnemonic, type).hasValue()) {
        printTo(parser.emitError(parser.getCurrentLocation()), "Unknown VPUIP Type '{0}'", mnemonic);
    }

    return type;
}

void vpux::VPUIP::VPUIPDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& os) const {
    VPUX_THROW_UNLESS(mlir::succeeded(generatedTypePrinter(type, os)), "Got unsupported Type : {}", type);
}

//
// DistributedBufferType::Accessors
//

vpux::ShapeRef vpux::VPUIP::DistributedBufferType::getShape() const {
    return vpux::ShapeRef(getImpl()->shape);
}

mlir::Type vpux::VPUIP::DistributedBufferType::getElementType() const {
    return getImpl()->elementType;
}

mlir::MemRefLayoutAttrInterface vpux::VPUIP::DistributedBufferType::getLayout() const {
    return getImpl()->layout;
}

vpux::IndexedSymbolAttr vpux::VPUIP::DistributedBufferType::getMemSpace() const {
    return getImpl()->memSpace;
}

VPU::DistributedTensorAttr vpux::VPUIP::DistributedBufferType::getDistribution() const {
    return getImpl()->distribution;
}

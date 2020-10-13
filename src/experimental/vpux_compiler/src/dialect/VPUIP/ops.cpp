//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

void vpux::VPUIP::VPUIPDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUIP/generated/ops.cpp.inc>
#undef GET_OP_LIST
            >();

    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPUIP/generated/types.cpp.inc>
#undef GET_TYPEDEF_LIST
            >();

    addAttributes<PhysicalProcessorAttr,
                  DMAEngineAttr,
                  PhysicalMemoryAttr,
                  ArchKindAttr,
                  MemoryLocationAttr,
                  ExecutionFlagAttr,
                  TaskTypeAttr>();
}

mlir::Attribute vpux::VPUIP::VPUIPDialect::parseAttribute(
        mlir::DialectAsmParser& parser,
        mlir::Type) const {
    StringRef mnenomic;
    if (mlir::failed(parser.parseKeyword(&mnenomic))) {
        printTo(parser.emitError(parser.getCurrentLocation()),
                "Failed to get VPUIP Attribute mnenomic");
        return nullptr;
    }

    if (mnenomic == PhysicalProcessorAttr::getMnemonic()) {
        return PhysicalProcessorAttr::parse(parser);
    } else if (mnenomic == DMAEngineAttr::getMnemonic()) {
        return DMAEngineAttr::parse(parser);
    } else if (mnenomic == PhysicalMemoryAttr::getMnemonic()) {
        return PhysicalMemoryAttr::parse(parser);
    } else if (mnenomic == ArchKindAttr::getMnemonic()) {
        return ArchKindAttr::parse(parser);
    } else if (mnenomic == MemoryLocationAttr::getMnemonic()) {
        return MemoryLocationAttr::parse(parser);
    } else if (mnenomic == ExecutionFlagAttr::getMnemonic()) {
        return ExecutionFlagAttr::parse(parser);
    } else if (mnenomic == TaskTypeAttr::getMnemonic()) {
        return TaskTypeAttr::parse(parser);
    }

    printTo(parser.emitError(parser.getCurrentLocation()),
            "Unknown VPUIP Attribute '{0}'",
            mnenomic);
    return nullptr;
}

void vpux::VPUIP::VPUIPDialect::printAttribute(
        mlir::Attribute attr,
        mlir::DialectAsmPrinter& os) const {
    llvm::TypeSwitch<mlir::Attribute>(attr)
            .Case<PhysicalProcessorAttr>([&os](PhysicalProcessorAttr proc) {
                proc.print(os);
            })
            .Case<DMAEngineAttr>([&os](DMAEngineAttr dma) {
                dma.print(os);
            })
            .Case<PhysicalMemoryAttr>([&os](PhysicalMemoryAttr mem) {
                mem.print(os);
            })
            .Case<ArchKindAttr>([&os](ArchKindAttr arch) {
                arch.print(os);
            })
            .Case<MemoryLocationAttr>([&os](MemoryLocationAttr attr) {
                attr.print(os);
            })
            .Case<ExecutionFlagAttr>([&os](ExecutionFlagAttr attr) {
                attr.print(os);
            })
            .Case<TaskTypeAttr>([&os](TaskTypeAttr attr) {
                attr.print(os);
            });
}

mlir::Type vpux::VPUIP::VPUIPDialect::parseType(
        mlir::DialectAsmParser& parser) const {
    StringRef mnemonic;
    if (mlir::failed(parser.parseKeyword(&mnemonic))) {
        printTo(parser.emitError(parser.getCurrentLocation()),
                "Failed to get VPUIP Type mnemonic");
        return nullptr;
    }

    const auto type = generatedTypeParser(getContext(), parser, mnemonic);

    if (type == nullptr) {
        printTo(parser.emitError(parser.getCurrentLocation()),
                "Unknown VPUIP Type '{0}'",
                mnemonic);
    }

    return type;
}

void vpux::VPUIP::VPUIPDialect::printType(mlir::Type type,
                                          mlir::DialectAsmPrinter& os) const {
    generatedTypePrinter(type, os);
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIP/generated/ops.cpp.inc>
#undef GET_OP_CLASSES

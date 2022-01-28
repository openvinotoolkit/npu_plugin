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

#include "vpux/compiler/dialect/VPU/types.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPU/generated/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// VPUDialect::registerTypes
//

void vpux::VPU::VPUDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPU/generated/types.cpp.inc>
            >();
}

//
// Dialect hooks
//

mlir::Type vpux::VPU::VPUDialect::parseType(mlir::DialectAsmParser& parser) const {
    StringRef mnemonic;
    if (mlir::failed(parser.parseKeyword(&mnemonic))) {
        printTo(parser.emitError(parser.getCurrentLocation()), "Failed to get VPU Type mnemonic");
        return nullptr;
    }

    mlir::Type type;
    if (!generatedTypeParser(parser, mnemonic, type).hasValue()) {
        printTo(parser.emitError(parser.getCurrentLocation()), "Unknown VPU Type '{0}'", mnemonic);
    }

    return type;
}

void vpux::VPU::VPUDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& os) const {
    VPUX_THROW_UNLESS(mlir::succeeded(generatedTypePrinter(type, os)), "Got unsupported Type : {0}", type);
}

//
// vpux::VPU::DistributedTensorType
//

void vpux::VPU::DistributedTensorType::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                                llvm::function_ref<void(Type)> walkTypesFn) const {
    walkTypesFn(getElementType());
    if (!getOrder().isIdentity()) {
        walkAttrsFn(getOrder());
    }
    walkAttrsFn(getMemSpace());
    walkAttrsFn(getDistribution());
}

void vpux::VPU::DistributedTensorType::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    for (auto& dim : getShape()) {
        printer << dim << "x";
    }
    printer << getElementType();

    const auto order = getOrder();
    if (!order.isIdentity()) {
        printer << ", " << order;
    }
    printer << ", " << getMemSpace();
    printer << ", " << getDistribution();
    printer << ">";
}

mlir::Type vpux::VPU::DistributedTensorType::parse(mlir::DialectAsmParser& parser) {
    if (parser.parseLess())
        return Type();

    SmallVector<int64_t> shape;

    int64_t dim = 0;
    while (parser.parseOptionalInteger(dim).hasValue() && parser.parseXInDimensionList().succeeded()) {
        shape.push_back(dim);
    }

    mlir::Type elemType;
    if (parser.parseType(elemType)) {
        return Type();
    }
    if (parser.parseComma()) {
        return Type();
    }

    mlir::AffineMapAttr order;
    if (parser.parseAttribute(order)) {
        return Type();
    } else {
        order = mlir::AffineMapAttr::get(DimsOrder::fromNumDims(shape.size()).toAffineMap(parser.getContext()));
    }
    if (parser.parseComma()) {
        return Type();
    }

    mlir::SymbolRefAttr memSpace;
    if (parser.parseAttribute(memSpace)) {
        return Type();
    }
    if (parser.parseComma()) {
        return Type();
    }

    DistributedTensorAttr distribution;
    if (parser.parseAttribute(distribution)) {
        return Type();
    }
    if (parser.parseGreater()) {
        return Type();
    }

    return get(parser.getContext(), makeArrayRef(shape), elemType, order, memSpace, distribution);
}

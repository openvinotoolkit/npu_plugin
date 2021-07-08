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

#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// ConvertElemTypeAttr::verify
//

mlir::LogicalResult vpux::Const::ConvertElemTypeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                             mlir::Type elemType) {
    if (elemType == nullptr) {
        return printTo(emitError(), "Got NULL 'elemType' in 'ConvertElemTypeAttr'");
    }

    if (!elemType.isIntOrFloat()) {
        return printTo(emitError(), "Only integers and floats are supported in ConvertElemTypeAttr");
    }

    return mlir::success();
}

//
// ConvertElemTypeAttr::print
//

void vpux::Const::ConvertElemTypeAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printType(getElemType());
    printer << ">";
}

//
// ConvertElemTypeAttr::parse
//

mlir::Attribute vpux::Const::ConvertElemTypeAttr::parse(mlir::MLIRContext*, mlir::DialectAsmParser& parser,
                                                        mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::Type elemType;
    if (mlir::failed(parser.parseType(elemType))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::ConvertElemTypeAttr>(elemType);
}

//
// ConvertElemTypeAttr::inferOutputType
//

mlir::ShapedType vpux::Const::ConvertElemTypeAttr::inferOutputType(mlir::ShapedType input) const {
    VPUX_THROW_UNLESS(input.getElementType().isIntOrFloat(), "Can't convert '{0}' element type to '{1}'",
                      input.getElementType(), getElemType());
    return input.clone(getElemType());
}

//
// ConvertElemTypeAttr::transform
//

Const::Content vpux::Const::ConvertElemTypeAttr::transform(vpux::Const::Content& input) const {
    return Const::Content::moveBuffer(inferOutputType(input.getType()), std::move(input));
}

//
// ContentAttr::convertElemType
//

Const::ContentAttr vpux::Const::ContentAttr::convertElemType(mlir::Type newElemType) const {
    return get(*this, Const::ConvertElemTypeAttr::get(newElemType).cast<Const::TransformAttrInterface>());
}

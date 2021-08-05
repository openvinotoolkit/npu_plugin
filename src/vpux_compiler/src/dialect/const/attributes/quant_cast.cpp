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
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// QuantCastAttr::verify
//

mlir::LogicalResult vpux::Const::QuantCastAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                       mlir::quant::QuantizedType elemType) {
    if (elemType == nullptr) {
        return printTo(emitError(), "Got NULL 'elemType' in 'QuantCastAttr'");
    }

    return mlir::success();
}

//
// QuantCastAttr::print
//

void vpux::Const::QuantCastAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printType(getElemType());
    printer << ">";
}

//
// QuantCastAttr::parse
//

mlir::Attribute vpux::Const::QuantCastAttr::parse(mlir::MLIRContext*, mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::quant::QuantizedType elemType;
    if (mlir::failed(parser.parseType(elemType))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::QuantCastAttr>(elemType);
}

//
// QuantCastAttr::inferOutputType
//

mlir::ShapedType vpux::Const::QuantCastAttr::inferOutputType(mlir::ShapedType input) const {
    const auto quantStorateType = normalizeQuantStorageType(getElemType().getStorageType());

    VPUX_THROW_UNLESS(input.getElementType() == quantStorateType, "Can't cast '{0}' element type to '{1}'",
                      input.getElementType(), getElemType());

    return changeElemType(input, getElemType());
}

//
// QuantCastAttr::transform
//

Const::Content vpux::Const::QuantCastAttr::transform(vpux::Const::Content& input) const {
    return Const::Content::moveBuffer(inferOutputType(input.getType()), std::move(input));
}

//
// ContentAttr::quantCast
//

Const::ContentAttr vpux::Const::ContentAttr::quantCast(mlir::quant::QuantizedType newElemType) const {
    return get(*this, Const::QuantCastAttr::get(newElemType).cast<Const::TransformAttrInterface>());
}

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
// ContentAttr::walkImmediateSubElements
//

void vpux::Const::QuantCastAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)>,
                                                          llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
    if (const auto elemTypeVal = getElemType()) {
        walkTypesFn(elemTypeVal);
    }
}

//
// QuantCastAttr::verify
//

mlir::LogicalResult vpux::Const::QuantCastAttr::verify(FuncRef<mlir::InFlightDiagnostic()>,
                                                       mlir::quant::QuantizedType) {
    return mlir::success();
}

//
// QuantCastAttr::print
//

void vpux::Const::QuantCastAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    if (const auto elemTypeVal = getElemType()) {
        printer.printType(elemTypeVal);
    }
    printer << ">";
}

//
// QuantCastAttr::parse
//

mlir::Attribute vpux::Const::QuantCastAttr::parse(mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::quant::QuantizedType elemType;
    parser.parseOptionalType(elemType);

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::QuantCastAttr>(parser.getContext(), elemType);
}

//
// QuantCastAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::QuantCastAttr::inferOutputType(vpux::NDTypeInterface input) const {
    if (const auto elemTypeVal = getElemType()) {
        const auto quantStorateType = normalizeQuantStorageType(elemTypeVal);

        VPUX_THROW_UNLESS(input.getElementType() == quantStorateType, "Can't cast '{0}' element type to '{1}'",
                          input.getElementType(), getElemType());

        return input.changeElemType(elemTypeVal);
    }

    const auto quantType = input.getElementType().dyn_cast<mlir::quant::QuantizedType>();
    VPUX_THROW_UNLESS(quantType != nullptr, "Unable to restore storage type from non-quantized type");

    return input.changeElemType(normalizeQuantStorageType(quantType));
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
    return get(*this, Const::QuantCastAttr::get(getContext(), newElemType).cast<Const::TransformAttrInterface>());
}

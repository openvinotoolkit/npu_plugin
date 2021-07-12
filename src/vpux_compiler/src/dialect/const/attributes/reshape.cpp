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
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// ReshapeAttr::verify
//

mlir::LogicalResult vpux::Const::ReshapeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::ArrayAttr shape) {
    if (shape == nullptr) {
        return printTo(emitError(), "Got NULL 'shape' in 'ReshapeAttr'");
    }

    for (const auto dimAttr : shape.getValue()) {
        if (!dimAttr.isa<mlir::IntegerAttr>()) {
            return printTo(emitError(), "Got non-integer value '{0}' in 'shape' for 'ReshapeAttr'", dimAttr);
        }
        if (dimAttr.cast<mlir::IntegerAttr>().getInt() <= 0) {
            return printTo(emitError(), "Got unsupported dimension value '{0}' in 'shape' for 'ReshapeAttr'", dimAttr);
        }
    }

    return mlir::success();
}

//
// ReshapeAttr::print
//

void vpux::Const::ReshapeAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getShape());
    printer << ">";
}

//
// ReshapeAttr::parse
//

mlir::Attribute vpux::Const::ReshapeAttr::parse(mlir::MLIRContext*, mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::ArrayAttr shape;
    if (mlir::failed(parser.parseAttribute(shape))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::ReshapeAttr>(shape);
}

//
// ReshapeAttr::inferOutputType
//

mlir::ShapedType vpux::Const::ReshapeAttr::inferOutputType(mlir::ShapedType input) const {
    const auto inOrder = DimsOrder::fromType(input);
    VPUX_THROW_UNLESS(inOrder == DimsOrder::fromNumDims(input.getRank()),
                      "Can't apply Reshape transformation to DimsOrder '{0}'", inOrder);

    const auto newShape = parseIntArrayAttr(getShape());
    return changeShape(input, ShapeRef(newShape));
}

//
// ReshapeAttr::transform
//

Const::Content vpux::Const::ReshapeAttr::transform(vpux::Const::Content& input) const {
    return Const::Content::moveBuffer(inferOutputType(input.getType()), std::move(input));
}

//
// ContentAttr::reshape
//

Const::ContentAttr vpux::Const::ContentAttr::reshape(ShapeRef newShape) const {
    return get(
            *this,
            Const::ReshapeAttr::get(getInt64ArrayAttr(getContext(), newShape)).cast<Const::TransformAttrInterface>());
}

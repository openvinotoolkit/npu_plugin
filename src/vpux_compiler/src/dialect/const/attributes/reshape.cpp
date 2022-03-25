//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// ReshapeAttr::walkImmediateSubElements
//

void vpux::Const::ReshapeAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                        llvm::function_ref<void(mlir::Type)>) const {
    walkAttrsFn(getShape());
}

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

mlir::Attribute vpux::Const::ReshapeAttr::parse(mlir::DialectAsmParser& parser, mlir::Type) {
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

vpux::NDTypeInterface vpux::Const::ReshapeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const Bit typeSizeInBits = input.getElemTypeSize();
    VPUX_THROW_UNLESS(typeSizeInBits.count() >= CHAR_BIT, "Got sub-byte input '{0}' in ReshapeAttr", typeSizeInBits);

    const auto inOrder = input.getDimsOrder();
    VPUX_THROW_UNLESS(inOrder == DimsOrder::fromNumDims(input.getRank()),
                      "Can't apply Reshape transformation to DimsOrder '{0}'", inOrder);

    const auto newShape = parseIntArrayAttr<int64_t>(getShape());
    return input.changeShape(ShapeRef(newShape));
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
    return get(*this,
               Const::ReshapeAttr::get(getIntArrayAttr(getContext(), newShape)).cast<Const::TransformAttrInterface>());
}

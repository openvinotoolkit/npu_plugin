//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/DialectImplementation.h>
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/IE/loop.hpp"

using namespace vpux;

//
// BroadcastAttr::walkImmediateSubElements
//

void vpux::Const::BroadcastAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                          llvm::function_ref<void(mlir::Type)>) const {
    walkAttrsFn(getAxis());
    walkAttrsFn(getValue());
}

//
// BroadcastAttr::replaceImmediateSubElements
//

mlir::Attribute vpux::Const::BroadcastAttr::replaceImmediateSubElements(ArrayRef<mlir::Attribute> replAttrs,
                                                                        ArrayRef<mlir::Type>) const {
    VPUX_THROW_WHEN(replAttrs.size() < 2, "Replace attrs array is too short: '{0}'", replAttrs.size());
    return get(replAttrs[0].dyn_cast_or_null<mlir::IntegerAttr>(), replAttrs[1].dyn_cast_or_null<mlir::IntegerAttr>());
}

//
// BroadcastAttr::print
//

void vpux::Const::BroadcastAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getAxis());
    printer << ", ";
    printer.printAttribute(getValue());
    printer << ">";
}

//
// PadWithZeroAttr::parse
//

mlir::Attribute vpux::Const::BroadcastAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::IntegerAttr axis;
    if (mlir::failed(parser.parseAttribute(axis))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::IntegerAttr value;
    if (mlir::failed(parser.parseAttribute(value))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::BroadcastAttr::get(axis, value);
}

//
// BroadcastAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::BroadcastAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const Bit typeSizeInBits = input.getElemTypeSize();
    VPUX_THROW_UNLESS(typeSizeInBits.count() >= CHAR_BIT, "Got sub-byte input '{0}' in BroadcastAttr",
                      input.getElementType());

    const auto value = getValue().getInt();
    const auto axis = Dim(getAxis().getInt());

    const auto inShape = input.getShape();

    VPUX_THROW_UNLESS(value >= inShape[axis],
                      "Value cannot be broadcasted due to new value's size is less than old one: {0} < {1}", value,
                      inShape[axis]);

    const Shape padBefore(inShape.size(), 0);

    Shape padAfter(inShape.size(), 0);
    padAfter[axis] = value - inShape[axis];

    return input.pad(padBefore, padAfter);
}

//
// BroadcastAttr::transform
//

Const::Content vpux::Const::BroadcastAttr::transform(vpux::Const::Content& input) const {
    VPUX_THROW_UNLESS(input.isSplat(), "Only splat constants might be broadcasted, for other cases use PadWithZero");

    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()), input.getStorageElemType(),
                                                  input.isSplat());

    const auto inBuf = input.getRawStorageBuf();
    auto outBuf = output.getRawTempBuf();

    std::copy_n(inBuf.data(), inBuf.size(), outBuf.data());

    return output;
}

Const::ContentAttr vpux::Const::ContentAttr::broadcast(Dim axis, int64_t value) const {
    return get(*this, Const::BroadcastAttr::get(getIntAttr(getContext(), axis.ind()), getIntAttr(getContext(), value))
                              .cast<Const::TransformAttrInterface>());
}

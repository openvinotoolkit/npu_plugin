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
// BroadcastAttr::print
//

void vpux::Const::BroadcastAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getAxis());
    printer << ", ";
    printer.printAttribute(getValue());
    printer << ">";
}

//
// PadWithZeroAttr::parse
//

mlir::Attribute vpux::Const::BroadcastAttr::parse(mlir::DialectAsmParser& parser, mlir::Type) {
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

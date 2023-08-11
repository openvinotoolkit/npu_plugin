//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// AddAttr::walkImmediateSubElements
//

void vpux::Const::AddAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                    llvm::function_ref<void(mlir::Type)>) const {
    walkAttrsFn(getBias());
}

//
// AddAttr::replaceImmediateSubElements
//

mlir::Attribute vpux::Const::AddAttr::replaceImmediateSubElements(ArrayRef<mlir::Attribute> replAttrs,
                                                                  ArrayRef<mlir::Type>) const {
    VPUX_THROW_WHEN(replAttrs.size() < 1, "Replace attrs array is too short: '{0}'", replAttrs.size());
    return get(replAttrs[0].dyn_cast_or_null<mlir::FloatAttr>());
}

//
// AddAttr::print
//

void vpux::Const::AddAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getBias());
    printer << ">";
}

//
// PadWithZeroAttr::parse
//

mlir::Attribute vpux::Const::AddAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::FloatAttr bias;
    if (mlir::failed(parser.parseAttribute(bias))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::AddAttr::get(bias);
}

//
// AddAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::AddAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const Bit typeSizeInBits = input.getElemTypeSize();
    VPUX_THROW_UNLESS(typeSizeInBits.count() >= CHAR_BIT, "Got sub-byte input '{0}' in AddAttr",
                      input.getElementType());

    return input;
}

//
// AddAttr::transform
//

Const::Content vpux::Const::AddAttr::transform(vpux::Const::Content& input) const {
    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()),
                                                  mlir::Float32Type::get(getContext()), input.isSplat());

    const auto values = input.getValues<float>();
    auto shiftedVals = output.getTempBuf<float>();

    const auto bias = static_cast<float>(getBias().getValue().convertToDouble());

    loop_1d(LoopExecPolicy::Parallel, shiftedVals.size(), [&](size_t i) {
        shiftedVals[i] = values[i] + bias;
    });

    return output;
}

Const::ContentAttr vpux::Const::ContentAttr::add(double bias) const {
    return get(*this, Const::AddAttr::get(getFPAttr(getContext(), bias)).cast<Const::TransformAttrInterface>());
}

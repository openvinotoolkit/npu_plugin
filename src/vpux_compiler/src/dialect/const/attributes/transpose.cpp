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
#include "vpux/compiler/dialect/const/utils/transformations.hpp"

#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// TransposeAttr::walkImmediateSubElements
//

void vpux::Const::TransposeAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                          llvm::function_ref<void(mlir::Type)>) const {
    walkAttrsFn(getOrder());
}

//
// TransposeAttr::verify
//

mlir::LogicalResult vpux::Const::TransposeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                       mlir::AffineMapAttr order) {
    if (order == nullptr) {
        return printTo(emitError(), "Got NULL 'order' in 'TransposeAttr'");
    }

    if (!order.getValue().isPermutation()) {
        return printTo(emitError(), "Got non permutation 'order' in 'TransposeAttr'");
    }

    return mlir::success();
}

//
// TransposeAttr::print
//

void vpux::Const::TransposeAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getOrder());
    printer << ">";
}

//
// TransposeAttr::parse
//

mlir::Attribute vpux::Const::TransposeAttr::parse(mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::AffineMapAttr order;
    if (mlir::failed(parser.parseAttribute(order))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::TransposeAttr>(order);
}

//
// TransposeAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::TransposeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const Bit typeSizeInBits = input.getElemTypeSize();
    VPUX_THROW_UNLESS(typeSizeInBits.count() >= CHAR_BIT, "Got sub-byte input '{0}' in TransposeAttr",
                      input.getElementType());

    const auto order = DimsOrder::fromAffineMap(getOrder().getValue());
    VPUX_THROW_UNLESS(order.numDims() == checked_cast<size_t>(input.getRank()),
                      "DimsOrder '{0}' doesn't match type '{1}'", order, input);

    const auto inputShape = input.getShape();
    Shape newShape(inputShape.size());
    for (size_t idx = 0; idx < newShape.size(); idx++) {
        newShape[Dim(idx)] = inputShape[order.dimAt(idx)];
    }

    return input.changeShape(newShape);
}

//
// TransposeAttr::transform
//

Const::Content vpux::Const::TransposeAttr::transform(vpux::Const::Content& input) const {
    // This is basically reorder with subsequent reshape.
    const auto outType = inferOutputType(input.getType());
    const auto inputOrder = input.getType().getDimsOrder();
    const auto inPerm = inputOrder.toAffineMap(getContext());
    const auto memPerm = inPerm.compose(getOrder().getValue());

    return Const::details::memPermuteTransformation(input, outType, memPerm);
}

//
// ContentAttr::transpose
//

Const::ContentAttr vpux::Const::ContentAttr::transpose(DimsOrder newOrder) const {
    return get(*this, Const::TransposeAttr::get(mlir::AffineMapAttr::get(newOrder.toAffineMap(getContext())))
                              .cast<Const::TransformAttrInterface>());
}

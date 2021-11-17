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
// ReorderAttr::walkImmediateSubElements
//

void vpux::Const::ReorderAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                        llvm::function_ref<void(mlir::Type)>) const {
    walkAttrsFn(getOrder());
}

//
// ReorderAttr::verify
//

mlir::LogicalResult vpux::Const::ReorderAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::AffineMapAttr order) {
    if (order == nullptr) {
        return printTo(emitError(), "Got NULL 'order' in 'ReorderAttr'");
    }

    if (!order.getValue().isPermutation()) {
        return printTo(emitError(), "Got non permutation 'order' in 'ReorderAttr'");
    }

    return mlir::success();
}

//
// ReorderAttr::print
//

void vpux::Const::ReorderAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getOrder());
    printer << ">";
}

//
// ReorderAttr::parse
//

mlir::Attribute vpux::Const::ReorderAttr::parse(mlir::DialectAsmParser& parser, mlir::Type) {
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

    return parser.getChecked<Const::ReorderAttr>(order);
}

//
// ReorderAttr::inferOutputType
//

mlir::ShapedType vpux::Const::ReorderAttr::inferOutputType(mlir::ShapedType input) const {
    const auto order = DimsOrder::fromAffineMap(getOrder().getValue());
    VPUX_THROW_UNLESS(order.numDims() == checked_cast<size_t>(input.getRank()),
                      "DimsOrder '{0}' doesn't match type '{1}'", order, input);

    return changeDimsOrder(input, order);
}

//
// ReorderAttr::transform
//

Const::Content vpux::Const::ReorderAttr::transform(vpux::Const::Content& input) const {
    const auto outType = inferOutputType(input.getType());
    const auto outOrder = DimsOrder::fromType(outType);
    return Const::details::reorderTransformation(input, outType, outOrder);
}

//
// ContentAttr::reorder
//

Const::ContentAttr vpux::Const::ContentAttr::reorder(DimsOrder newOrder) const {
    return get(*this, Const::ReorderAttr::get(mlir::AffineMapAttr::get(newOrder.toAffineMap(getContext())))
                              .cast<Const::TransformAttrInterface>());
}

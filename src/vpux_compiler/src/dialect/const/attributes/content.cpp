//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/const_logger.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <exception>
#include <numeric>

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/const/attributes.cpp.inc>

//
// ConstDialect::initialize
//

void vpux::Const::ConstDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/const/ops.cpp.inc>
            >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/const/attributes.cpp.inc>
            >();
}

//
// ContentAttr::walkImmediateSubElements
//

void vpux::Const::ContentAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                        llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
    if (auto content = getBaseContent()) {
        walkAttrsFn(content);
    }

    if (auto transforms = getImpl()->transformations) {
        walkAttrsFn(transforms);
    }

    walkTypesFn(getType());
}

//
// ContentAttr::replaceImmediateSubElements
//

mlir::Attribute vpux::Const::ContentAttr::replaceImmediateSubElements(ArrayRef<mlir::Attribute> replAttrs,
                                                                      ArrayRef<mlir::Type> replTypes) const {
    VPUX_THROW_WHEN(replAttrs.size() < 2, "Replace attrs array is too short: '{0}'", replAttrs.size());
    VPUX_THROW_WHEN(replTypes.size() < 1, "Replace types array is too short: '{0}'", replTypes.size());
    return get(replAttrs[0].dyn_cast_or_null<mlir::ElementsAttr>(), replAttrs[1].dyn_cast_or_null<mlir::ArrayAttr>(),
               replTypes[0]);
}

//
// ContentAttr::verify
//

mlir::LogicalResult vpux::Const::ContentAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::ElementsAttr baseContent, mlir::ArrayAttr transformations,
                                                     vpux::NDTypeInterface finalType) {
    if (baseContent == nullptr) {
        return printTo(emitError(), "Got NULL 'baseContent' in 'ContentAttr'");
    }

    if (!baseContent.getType().getElementType().isIntOrFloat()) {
        return printTo(emitError(), "Got unsupported element type for 'baseContent' in 'ContentAttr' : '{0}'",
                       baseContent.getType().getElementType());
    }

    if (baseContent.isa<mlir::DenseElementsAttr>()) {
        // OK
    } else if (const auto opaque = baseContent.dyn_cast<Const::OpaqueElementsAttr>()) {
        const size_t numElems = opaque.getNumElements();
        const Byte elemTypeSize = vpux::getElemTypeSize(opaque.getType());

        const auto bytes = opaque.getValue();

        if (bytes.size() != numElems * elemTypeSize.count()) {
            return printTo(emitError(),
                           "Size of opaque buffer '{0}' in 'OpaqueElementsAttr' doesn't match its Type '{1}'",
                           bytes.size(), opaque.getType());
        }
    } else {
        return printTo(emitError(), "Got unsupported 'baseContent' in 'ContentAttr'");
    }

    if (transformations != nullptr) {
        const auto transormationList = transformations.getValue();
        auto inferedFinalType = baseContent.getType().cast<vpux::NDTypeInterface>();

        for (const auto attr : transormationList) {
            const auto trAttr = attr.dyn_cast<Const::TransformAttrInterface>();
            if (trAttr == nullptr) {
                return printTo(emitError(), "Got non transformation attribute : '{0}'", attr);
            }

            inferedFinalType = trAttr.inferOutputType(inferedFinalType);
        }

        if (finalType != inferedFinalType) {
            return printTo(emitError(), "Final type '{0}' doesn't match inferred type '{1}'", finalType,
                           inferedFinalType);
        }
    }

    return mlir::success();
}

//
// wrapBaseContent
//

namespace {

Const::Content wrapBaseContent(mlir::ElementsAttr baseContent) {
    ArrayRef<char> data;
    bool isSplat = false;

    if (const auto dense = baseContent.dyn_cast<mlir::DenseElementsAttr>()) {
        data = dense.getRawData();
        isSplat = dense.isSplat();
    } else {
        const auto opaque = baseContent.cast<Const::OpaqueElementsAttr>();
        const auto bytes = opaque.getValue();
        data = makeArrayRef(bytes.data(), bytes.size());

        VPUX_THROW_UNLESS(mlir::DenseElementsAttr::isValidRawBuffer(baseContent.getType(), data, isSplat),
                          "Got invalid opaque buffer");
    }

    return Const::Content::fromRawBuffer(baseContent.getType().cast<vpux::NDTypeInterface>(), data,
                                         baseContent.getType().getElementType(), isSplat);
}

}  // namespace

//
// ContentAttr::fold
//

Const::Content vpux::Const::ContentAttr::fold() const {
    auto res = wrapBaseContent(getBaseContent());

    if (const auto transformations = getImpl()->transformations) {
        for (const auto attr : transformations.getAsRange<Const::TransformAttrInterface>()) {
            Const::logger().trace("Applying transformation: {0}", attr);
            res = attr.transform(res);
        }
    }

    return res;
}

//
// ContentAttr::getBaseContent
//

mlir::ElementsAttr vpux::Const::ContentAttr::getBaseContent() const {
    return getImpl()->baseContent;
}

//
// ContentAttr::getTransformations
//

SmallVector<Const::TransformAttrInterface> vpux::Const::ContentAttr::getTransformations() const {
    if (const auto transformations = getImpl()->transformations) {
        return to_small_vector(transformations.getAsRange<vpux::Const::TransformAttrInterface>());
    }

    return {};
}

//
// ContentAttr::getType
//

vpux::NDTypeInterface vpux::Const::ContentAttr::getType() const {
    return getImpl()->finalType;
}

//
// ContentAttr::print
//

void vpux::Const::ContentAttr::print(mlir::AsmPrinter& printer) const {
    printer.printAttribute(getBaseContent());
    if (const auto transformations = getImpl()->transformations) {
        printer << ", ";
        printer.printAttribute(transformations);
    }
}

//
// ContentAttr::parse
//

mlir::Attribute vpux::Const::ContentAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    mlir::ElementsAttr baseContent;
    if (mlir::failed(parser.parseAttribute(baseContent))) {
        return nullptr;
    }

    mlir::ArrayAttr transformations;
    if (mlir::succeeded(parser.parseOptionalComma())) {
        if (mlir::failed(parser.parseAttribute(transformations))) {
            return nullptr;
        }
    }

    auto finalType = baseContent.getType().cast<vpux::NDTypeInterface>();

    if (transformations != nullptr) {
        for (const auto attr : transformations.getValue()) {
            const auto trAttr = attr.dyn_cast<Const::TransformAttrInterface>();
            VPUX_THROW_UNLESS(trAttr != nullptr, "Got non TransformAttr");

            finalType = trAttr.inferOutputType(finalType);
        }
    }

    return parser.getChecked<Const::ContentAttr>(baseContent, transformations, finalType);
}

// The list of tranformations can have the following position requirements if all types are present:
//   [NONE]* -> [PREFERRED_LAST]* -> [LAST]
// No two transformations with the LAST requirement can exist.
// The order of elements with the same requirement is stable.
Const::TransformAttrInterface* getInsertionPosition(SmallVector<Const::TransformAttrInterface>& transformations,
                                                    Const::TransformAttrInterface newTransformation) {
    auto endPosition = transformations.end();
    if (transformations.empty()) {
        return endPosition;
    }

    const auto lastTransformation = transformations.back();

    const auto newTransformationReq = newTransformation.getPositionRequirement();
    const auto lastTransformationReq = lastTransformation.getPositionRequirement();

    const auto newTransformationReqLast = newTransformationReq == vpux::Const::details::PositionRequirement::LAST;
    const auto lastTransformationReqLast = lastTransformationReq == vpux::Const::details::PositionRequirement::LAST;
    VPUX_THROW_WHEN(newTransformationReqLast && lastTransformationReqLast,
                    "Existing transformation with LAST position requirement");

    if (newTransformationReq == vpux::Const::details::PositionRequirement::NONE) {
        for (auto it = transformations.end(); it > transformations.begin(); --it) {
            if ((it - 1)->getPositionRequirement() == vpux::Const::details::PositionRequirement::NONE) {
                return it;
            }
        }
    }

    if (lastTransformationReqLast) {
        return endPosition - 1;
    }

    return endPosition;
}

//
// ContentAttr::addTransformation
//

Const::ContentAttr Const::ContentAttr::addTransformation(Const::ContentAttr input,
                                                         Const::TransformAttrInterface newTransformation) {
    auto transformations = input.getTransformations();
    auto insertionPosition = getInsertionPosition(transformations, newTransformation);
    insertionPosition = transformations.insert(insertionPosition, newTransformation);

    // Update all transformations attributes starting from inserted transformation
    auto baseContent = input.getBaseContent();
    for (auto it = insertionPosition; it != transformations.end(); it++) {
        SmallVector<mlir::Attribute> prevTransformations(transformations.begin(), it);
        auto updatedTransformation = it->updateAttributes(baseContent, prevTransformations);
        if (updatedTransformation != nullptr) {
            *it = updatedTransformation;
        }
    }

    auto newOutputType = baseContent.getType().cast<vpux::NDTypeInterface>();
    for (auto tr : transformations) {
        newOutputType = tr.inferOutputType(newOutputType);
    }

    const auto transformationsRaw = to_small_vector(
            transformations | transformed([](vpux::Const::TransformAttrInterface attr) -> mlir::Attribute {
                return attr;
            }));
    const auto transformationsAttr = mlir::ArrayAttr::get(input.getContext(), transformationsRaw);

    return Base::get(input.getContext(), input.getBaseContent(), transformationsAttr, newOutputType);
}

int64_t Const::ContentAttr::getNumberOfElements() const {
    auto transformations = getTransformations();
    if (!transformations.empty()) {
        if (auto sparsifyAttr = transformations.back().dyn_cast_or_null<Const::SparsifyAttr>()) {
            return Const::SparsifyAttr::countNumTotalElements(sparsifyAttr);
        }
    }
    return getType().getShape().totalSize();
}

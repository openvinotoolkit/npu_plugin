// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/sparsity.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <numeric>

using namespace vpux;

//
// ContentAttr::verify
//

mlir::LogicalResult vpux::Const::SparsifyAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                      mlir::BoolAttr compressOutputType,
                                                      mlir::ElementsAttr numActualElements) {
    if (compressOutputType == nullptr) {
        return printTo(emitError(), "Got NULL 'compressOutputType' in 'SparsifyAttr'");
    }
    if (numActualElements != nullptr) {
        if (!numActualElements.getType().getElementType().isIntOrIndex()) {
            return printTo(emitError(), "Got unsupported 'numActualElements' in 'SparsifyAttr' : '{0}'",
                           numActualElements.getType().getElementType());
        }
        if (!numActualElements.isa<mlir::DenseElementsAttr>()) {
            return printTo(emitError(), "Got unsupported 'numActualElements' in 'SparsifyAttr'");
        }
        auto values = numActualElements.getValues<int64_t>();
        for (const auto value : values) {
            if (value < 0) {
                return printTo(emitError(),
                               "Got negative elem number value '{0}' in 'numActualElements' for 'SparsifyAttr'", value);
            }
        }
    }

    return mlir::success();
}

//
// SparsifyAttr::print
//

void vpux::Const::SparsifyAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getCompressOutputType());
    if (getNumActualElements() != nullptr) {
        printer << ", ";
        printer.printAttribute(getNumActualElements());
    }
    printer << ">";
}

//
// SparsifyAttr::parse
//

mlir::Attribute vpux::Const::SparsifyAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::BoolAttr compressOutputType;
    if (mlir::failed(parser.parseAttribute(compressOutputType))) {
        return nullptr;
    }

    mlir::ElementsAttr numActualElements;
    if (mlir::succeeded(parser.parseOptionalComma())) {
        if (mlir::failed(parser.parseAttribute(numActualElements))) {
            return nullptr;
        }
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::SparsifyAttr>(compressOutputType, numActualElements);
}

//
// SparsifyAttr::inferOutputType
//

vpux::NDTypeInterface compressType(vpux::NDTypeInterface inputType, mlir::ElementsAttr numElemsAttr) {
    const auto elemByteSize = getElemTypeSize(inputType).to<Byte>().count();
    const auto numElems = numElemsAttr.getValues<int64_t>();
    int64_t totalByteSize = 0;
    for (auto num : numElems) {
        totalByteSize += alignValUp<int64_t>(num * elemByteSize, VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT);
    }
    const auto newShape = Shape({totalByteSize, 1, 1, 1});
    return inputType.changeShapeElemType(newShape, getUInt8Type(inputType.getContext()));
}

vpux::NDTypeInterface vpux::Const::SparsifyAttr::inferOutputType(vpux::NDTypeInterface inputType) const {
    if (getCompressOutputType().getValue() != false) {
        return compressType(inputType, getNumActualElements());
    }
    return inputType;
}

namespace {

template <typename StorageType>
Const::Content sparsify(const Const::Content& content, int64_t sparsifyValue, NDTypeInterface inputType,
                        NDTypeInterface outputType) {
    auto output = Const::Content::allocTempBuffer(outputType, outputType.getElementType(), false);
    output.fillWithZero();
    auto outBuf = output.getRawTempBuf();
    auto outBlobPtr = reinterpret_cast<StorageType*>(outBuf.data());

    auto inputValues = content.getValues<StorageType>();

    auto inputShape = inputType.getShape();
    VPUX_THROW_UNLESS(inputShape.size() == 4, "Expected 4D input shape. Got {0}", inputShape);

    const auto OC = inputShape[Dims4D::Filter::OC];
    const auto IC = inputShape[Dims4D::Filter::IC];
    const auto KY = inputShape[Dims4D::Filter::KY];
    const auto KX = inputShape[Dims4D::Filter::KX];
    const auto workloadSize = IC * KY * KX;

    const auto castedSparsifyValue = checked_cast<StorageType>(sparsifyValue);
    constexpr auto byteSize = sizeof(castedSparsifyValue);

    int64_t outputIndex = 0;
    for (int64_t oc = 0; oc < OC; ++oc) {
        auto begin = oc * workloadSize;
        auto end = (oc + 1) * workloadSize;
        for (auto inputIndex = begin; inputIndex < end; ++inputIndex) {
            const auto inputValue = inputValues[inputIndex];
            if (inputValue == castedSparsifyValue) {
                continue;
            }
            outBlobPtr[outputIndex++] = inputValue;
        }
        const auto outputIndexByte = outputIndex * byteSize;
        if (outputIndexByte % VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT != 0) {
            const auto alignedOutputIndexByte =
                    alignValUp<int64_t>(outputIndexByte, VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT);
            outputIndex = alignedOutputIndexByte / byteSize;
        }
    }
    return output;
}

}  // namespace

//
// SparsifyAttr::transform
//

Const::Content Const::SparsifyAttr::transform(Const::Content& input) const {
    auto inputType = input.getType();
    auto outputType = compressType(inputType, getNumActualElements());

    auto inputElementType = inputType.getElementType();
    int64_t sparsifyValue = getSparsifyValue(inputElementType);

    if (inputElementType.isSignedInteger(8)) {
        return sparsify<int8_t>(input, sparsifyValue, inputType, outputType);
    } else if (inputElementType.isUnsignedInteger(8)) {
        return sparsify<uint8_t>(input, sparsifyValue, inputType, outputType);
    } else if (inputElementType.isF16()) {
        return sparsify<float16>(input, sparsifyValue, inputType, outputType);
    } else if (inputElementType.isBF16()) {
        return sparsify<bfloat16>(input, sparsifyValue, inputType, outputType);
    } else if (inputElementType.isF32()) {
        return sparsify<float>(input, sparsifyValue, inputType, outputType);
    }
    VPUX_THROW("Unexpected weights data type: {0}", inputElementType);
}

//
// SparsifyAttr::getPositionRequirement
//

Const::details::PositionRequirement Const::SparsifyAttr::getPositionRequirement() const {
    return Const::details::PositionRequirement::PREFERRED_LAST;
}

Const::TransformAttrInterface Const::SparsifyAttr::updateAttributes(mlir::ElementsAttr& baseContent,
                                                                    ArrayRef<mlir::Attribute> transformations) const {
    // Apply transformations
    auto outputType = baseContent.getType().cast<vpux::NDTypeInterface>();
    for (auto transformationAttr : transformations) {
        auto transformation = transformationAttr.dyn_cast<Const::TransformAttrInterface>();
        if (transformation != nullptr) {
            outputType = transformation.inferOutputType(outputType);
        }
    }

    const auto transformationsAttr = mlir::ArrayAttr::get(getContext(), transformations);
    auto contentAttr = ContentAttr::get(baseContent, transformationsAttr, outputType);
    auto content = contentAttr.fold();
    auto elementType = outputType.getElementType();

    auto numActualElements = vpux::getNumActualElements(content, elementType);

    const auto numActualElementsType =
            mlir::RankedTensorType::get({static_cast<int64_t>(numActualElements.size())}, getInt64Type(getContext()));
    const auto actualElems = mlir::DenseElementsAttr::get(numActualElementsType, makeArrayRef(numActualElements));

    auto updatedAttr = Const::SparsifyAttr::get(this->getCompressOutputType(), actualElems);
    return updatedAttr;
}

//
// SparsifyAttr::countNumTotalElements
//

int64_t vpux::Const::SparsifyAttr::countNumTotalElements(vpux::Const::SparsifyAttr transformation) {
    const auto numActualElementsAttr = transformation.getNumActualElements();
    VPUX_THROW_UNLESS(numActualElementsAttr != nullptr, "Sparsify transformation missing number of elements");
    const auto numActualElementsPerOC = numActualElementsAttr.getValues<int64_t>();
    return std::accumulate(numActualElementsPerOC.begin(), numActualElementsPerOC.end(), static_cast<int64_t>(0));
}

//
// ContentAttr::sparsify
//

Const::ContentAttr vpux::Const::ContentAttr::sparsify(bool compressOutputType) const {
    return get(*this, Const::SparsifyAttr::get(mlir::BoolAttr::get(getContext(), compressOutputType)));
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// RelocateWeightsTableAttr::walkImmediateSubElements
//

void vpux::Const::RelocateWeightsTableAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                                     llvm::function_ref<void(mlir::Type)>) const {
    walkAttrsFn(getWeightsPtr());
    walkAttrsFn(getSparsityPtr());
    walkAttrsFn(getOffsets());
}

//
// RelocateWeightsTableAttr::print
//

void vpux::Const::RelocateWeightsTableAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getWeightsPtr());
    printer << ", ";
    printer.printAttribute(getSparsityPtr());
    printer << ", ";
    printer.printAttribute(getOffsets());
    if (getWeightsElemByteSize() != nullptr) {
        printer << ", ";
        printer.printAttribute(getWeightsElemByteSize());
    }
    if (getWeightsCompression() != nullptr) {
        printer << ", ";
        printer.printAttribute(getWeightsCompression());
    }
    printer << ">";
}

//
// RelocateWeightsTableAttr::parse
//

mlir::Attribute vpux::Const::RelocateWeightsTableAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::IntegerAttr weightsPtr;
    mlir::IntegerAttr sparsityPtr;
    mlir::ArrayAttr offsets;
    mlir::IntegerAttr weightsElemByteSize;
    VPUIP::CompressionSchemeAttr weightsCompression;

    if (mlir::failed(parser.parseAttribute(weightsPtr))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    if (mlir::failed(parser.parseAttribute(sparsityPtr))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    if (mlir::failed(parser.parseAttribute(offsets))) {
        return nullptr;
    }

    if (mlir::succeeded(parser.parseGreater())) {
        return Const::RelocateWeightsTableAttr::get(weightsPtr, sparsityPtr, offsets, weightsElemByteSize,
                                                    weightsCompression);
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    if (mlir::failed(parser.parseAttribute(weightsElemByteSize))) {
        return nullptr;
    }

    if (mlir::succeeded(parser.parseGreater())) {
        return Const::RelocateWeightsTableAttr::get(weightsPtr, sparsityPtr, offsets, weightsElemByteSize,
                                                    weightsCompression);
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    if (mlir::failed(parser.parseAttribute(weightsCompression))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::RelocateWeightsTableAttr::get(weightsPtr, sparsityPtr, offsets, weightsElemByteSize,
                                                weightsCompression);
}

//
// RelocateWeightsTableAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::RelocateWeightsTableAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input;
}

//
// RelocateWeightsTableAttr::transform
//

Const::Content vpux::Const::RelocateWeightsTableAttr::transform(vpux::Const::Content& input) const {
    constexpr auto numElemPerOC = static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

    auto output =
            Const::Content::allocTempBuffer(inferOutputType(input.getType()),
                                            mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Signed), false);

    const auto values = input.getValues<int32_t>();
    auto patchedValues = output.getTempBuf<int32_t>();

    const auto weightsPtr = static_cast<int32_t>(*getWeightsPtr().getValue().getRawData());
    const auto sparsityPtr = static_cast<int32_t>(*getSparsityPtr().getValue().getRawData());
    const auto offsets = parseIntArrayAttr<int64_t>(getOffsets());

    int32_t weightPtrStep = 0;
    int32_t sparsityPtrStep = 0;
    if (values.size() >= numElemPerOC * 2) {
        weightPtrStep = values[1 * numElemPerOC + 0] - values[0 * numElemPerOC + 0];
        sparsityPtrStep = values[1 * numElemPerOC + 1] - values[0 * numElemPerOC + 1];
    }

    const int64_t OC = checked_cast<int64_t>(values.size() / numElemPerOC);
    const int64_t numClusters = checked_cast<int64_t>(offsets.size());

    SmallVector<int64_t> weightsPtrSteps(OC);
    if (getWeightsCompression() != nullptr) {
        const auto numElems = to_small_vector(getWeightsCompression().getNumElems().getValues<int64_t>());
        VPUX_THROW_UNLESS(numElems.size() == static_cast<size_t>(OC),
                          "Invalid weights compression with {0} elements for {1} channels", numElems.size(), OC);
        VPUX_THROW_UNLESS(getWeightsElemByteSize() != nullptr, "Missing weights element type attribute");
        const auto weightsElemByteSize = getWeightsElemByteSize().getInt();
        const auto alignment = (getWeightsCompression().getAlignment() != nullptr)
                                       ? getWeightsCompression().getAlignment().getInt()
                                       : VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT;

        int64_t weightsPtrOffset = 0;
        for (int64_t oc = 0, clusterIdx = 0; oc < OC; ++oc) {
            if ((clusterIdx + 1) < numClusters && oc >= offsets[clusterIdx + 1]) {
                clusterIdx++;
                weightsPtrOffset = 0;
            }
            weightsPtrSteps[oc] = weightsPtrOffset;
            const auto weightSetSize = (numElems[oc] * weightsElemByteSize);
            weightsPtrOffset += alignVal<int64_t>(weightSetSize, alignment);
        }
    } else {
        for (int64_t oc = 0, clusterIdx = 0; oc < OC; ++oc) {
            if ((clusterIdx + 1) < numClusters && oc >= offsets[clusterIdx + 1]) {
                clusterIdx++;
            }
            weightsPtrSteps[oc] = weightPtrStep * (oc - offsets[clusterIdx]);
        }
    }

    for (int64_t oc = 0, clusterIdx = 0; oc < OC; ++oc) {
        if ((clusterIdx + 1) < numClusters && oc >= offsets[clusterIdx + 1]) {
            clusterIdx++;
        }

        const auto wtInd = oc * numElemPerOC;

        patchedValues[wtInd + 0] = checked_cast<int32_t>(weightsPtr + weightsPtrSteps[oc]);

        patchedValues[wtInd + 1] = values[wtInd + 1];
        if (values[wtInd + 1] != VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY) {
            patchedValues[wtInd + 1] =
                    checked_cast<int32_t>(sparsityPtr + (oc - offsets[clusterIdx]) * sparsityPtrStep);
        }

        patchedValues[wtInd + 2] = values[wtInd + 2];
        patchedValues[wtInd + 3] = values[wtInd + 3];
    }

    return output;
}

Const::ContentAttr vpux::Const::ContentAttr::relocateWeightsTablePointers(
        uint64_t weightsPtr, uint64_t sparsityPtr, ShapeRef offsets, uint64_t weightsElemByteSize,
        VPUIP::CompressionSchemeAttr weightsCompression) const {
    return get(*this, Const::RelocateWeightsTableAttr::get(
                              getIntAttr(getContext(), weightsPtr), getIntAttr(getContext(), sparsityPtr),
                              getIntArrayAttr(getContext(), offsets), getIntAttr(getContext(), weightsElemByteSize),
                              weightsCompression)
                              .cast<Const::TransformAttrInterface>());
}

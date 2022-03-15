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

#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/func_ref.hpp"

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

void vpux::Const::RelocateWeightsTableAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getWeightsPtr());
    printer << ", ";
    printer.printAttribute(getSparsityPtr());
    printer << ", ";
    printer.printAttribute(getOffsets());
    printer << ">";
}

//
// RelocateWeightsTableAttr::parse
//

mlir::Attribute vpux::Const::RelocateWeightsTableAttr::parse(mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::IntegerAttr weightsPtr;
    if (mlir::failed(parser.parseAttribute(weightsPtr))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::IntegerAttr sparsityPtr;
    if (mlir::failed(parser.parseAttribute(sparsityPtr))) {
        return nullptr;
    }

    mlir::ArrayAttr offsets;
    if (mlir::failed(parser.parseAttribute(offsets))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::RelocateWeightsTableAttr::get(weightsPtr, sparsityPtr, offsets);
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

    const auto channelNum = values.size() / numElemPerOC;

    auto clusterIndex = 0;
    for (auto channelIndex = 0; channelIndex < checked_cast<int32_t>(channelNum); channelIndex++) {
        const auto wtInd = channelIndex * numElemPerOC;

        auto nextClusterIndex = clusterIndex + 1;
        if (checked_cast<size_t>(nextClusterIndex) < offsets.size() && channelIndex >= offsets[nextClusterIndex]) {
            clusterIndex++;
        }

        patchedValues[wtInd + 0] =
                checked_cast<int32_t>(weightsPtr + (channelIndex - offsets[clusterIndex]) * weightPtrStep);

        patchedValues[wtInd + 1] = values[wtInd + 1];
        if (values[wtInd + 1] != VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY) {
            patchedValues[wtInd + 1] =
                    checked_cast<int32_t>(sparsityPtr + (channelIndex - offsets[clusterIndex]) * sparsityPtrStep);
        }

        patchedValues[wtInd + 2] = values[wtInd + 2];
        patchedValues[wtInd + 3] = values[wtInd + 3];
    }

    return output;
}

Const::ContentAttr vpux::Const::ContentAttr::relocateWeightsTablePointers(uint64_t weightsPtr, uint64_t sparsityPtr,
                                                                          ShapeRef offsets) const {
    return get(*this, Const::RelocateWeightsTableAttr::get(getIntAttr(getContext(), weightsPtr),
                                                           getIntAttr(getContext(), sparsityPtr),
                                                           getIntArrayAttr(getContext(), offsets))
                              .cast<Const::TransformAttrInterface>());
}

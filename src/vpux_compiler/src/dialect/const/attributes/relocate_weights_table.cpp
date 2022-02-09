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

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// RelocateWeightsTableAttr::walkImmediateSubElements
//

void vpux::Const::RelocateWeightsTableAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                                     llvm::function_ref<void(mlir::Type)>) const {
    walkAttrsFn(getWeightsPtr());
    walkAttrsFn(getSparsityPtr());
}

//
// RelocateWeightsTableAttr::print
//

void vpux::Const::RelocateWeightsTableAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getWeightsPtr());
    printer << ", ";
    printer.printAttribute(getSparsityPtr());
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

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::RelocateWeightsTableAttr::get(weightsPtr, sparsityPtr);
}

//
// RelocateWeightsTableAttr::inferOutputType
//

mlir::ShapedType vpux::Const::RelocateWeightsTableAttr::inferOutputType(mlir::ShapedType input) const {
    return input;
}

//
// RelocateWeightsTableAttr::transform
//

Const::Content vpux::Const::RelocateWeightsTableAttr::transform(vpux::Const::Content& input) const {
    auto output =
            Const::Content::allocTempBuffer(inferOutputType(input.getType()),
                                            mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Signed), false);

    const auto values = input.getValues<int32_t>();
    auto patchedValues = output.getTempBuf<int32_t>();

    const auto weightsPtr = static_cast<int32_t>(*getWeightsPtr().getValue().getRawData());
    const auto sparsityPtr = static_cast<int32_t>(*getSparsityPtr().getValue().getRawData());
    const auto numElemPerOC = static_cast<size_t>(vpux::VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);
    loop_1d(LoopExecPolicy::Parallel, values.size() / numElemPerOC, [&](size_t i) {
        const auto wtInd = i * numElemPerOC;
        patchedValues[wtInd + 0] = weightsPtr + values[wtInd + 0];
        patchedValues[wtInd + 1] = values[wtInd + 1];
        if (values[wtInd + 1] != VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARISTY) {
            patchedValues[wtInd + 1] += sparsityPtr;
        }
        patchedValues[wtInd + 2] = values[wtInd + 2];
        patchedValues[wtInd + 3] = values[wtInd + 3];
    });

    return output;
}

Const::ContentAttr vpux::Const::ContentAttr::relocateWeightsTablePointers(uint64_t weightsPtr,
                                                                          uint64_t sparsityPtr) const {
    return get(*this, Const::RelocateWeightsTableAttr::get(getIntAttr(getContext(), weightsPtr),
                                                           getIntAttr(getContext(), sparsityPtr))
                              .cast<Const::TransformAttrInterface>());
}

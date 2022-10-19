//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

namespace {

mlir::Type inferElemType(IE::ReorderOpAdaptor reorder, mlir::Type inputElemType, mlir::MLIRContext* ctx) {
    const auto perAxisQType = inputElemType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();
    if (perAxisQType == nullptr) {
        return inputElemType;
    }

    const auto inputAxis = perAxisQType.getQuantizedDimension();

    const auto inOrder = DimsOrder::fromValue(reorder.input());
    const auto outOrder = DimsOrder::fromAffineMap(reorder.dstOrder().getValue());
    const auto memPerm = vpux::getPermutationFromOrders(inOrder, outOrder, ctx);

    const auto outAxis = DimsOrder::fromAffineMap(memPerm).toPermutation()[inputAxis];

    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(),
            perAxisQType.getScales(), perAxisQType.getZeroPoints(), outAxis.ind(), perAxisQType.getStorageTypeMin(),
            perAxisQType.getStorageTypeMax());
}

}  // namespace

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ReorderOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ReorderOpAdaptor reorder(operands, attrs);
    if (mlir::failed(reorder.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = reorder.input().getType().cast<mlir::RankedTensorType>();

    const auto outDesc = IE::getTensorAttr(reorder.dstOrder(), nullptr);

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

//
// inferElemTypeInfo
//

void vpux::IE::ReorderOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    auto outputElemType = inferElemType(*this, info.getInput(0), (*this)->getContext());

    for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
        info.setOutput(outputInd, outputElemType);
    }
}

void vpux::IE::ReorderOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    const auto outputElemType = info.getOutput(0);

    if (outputElemType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>() != nullptr) {
        // E#31029: implement propagate type up for per channel, currently it leads to failures in later passes.
        return;
    }

    for (size_t inputInd = 0; inputInd < info.getNumInputs(); ++inputInd) {
        info.setInput(inputInd, outputElemType);
    }
}

//
// Canonicalization
//

namespace {

#include <vpux/compiler/dialect/IE/rewriters/generated/reorder.hpp.inc>

}  // namespace

void vpux::IE::ReorderOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext*) {
    populateWithGenerated(patterns);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReorderOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    if (const auto cst = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return cst.reorder(DimsOrder::fromValue(output()));
    }

    return nullptr;
}

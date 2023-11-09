//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dilated_utils.hpp"

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ExpandDilatedOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ExpandDilatedOpAdaptor expandDilated(operands, attrs);
    if (mlir::failed(expandDilated.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = expandDilated.input().getType().dyn_cast<vpux::NDTypeInterface>();
    if (!inType) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(expandDilated.dilations());
    const auto newType = getDilatedType(inType, ShapeRef(dilations)).cast<mlir::RankedTensorType>();
    inferredReturnShapes.emplace_back(newType.getShape(), newType.getElementType(), newType.getEncoding());

    return mlir::success();
}

//
// inferElemTypeInfo
//

void vpux::IE::ExpandDilatedOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    const auto inputElemType = info.getInput(0);
    const auto dilationsVal = parseIntArrayAttr<int64_t>(dilations());
    if (inputElemType.isa<mlir::quant::UniformQuantizedPerAxisType>() && dilationsVal.size() > 2) {
        // E#84659: implement propagate type up for per channel, currently it leads to failures in later passes.
        return;
    }

    for (size_t outputInd = 0; outputInd < info.getNumOutputs(); ++outputInd) {
        info.setOutput(outputInd, inputElemType);
    }
}

void vpux::IE::ExpandDilatedOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    const auto outputElemType = info.getOutput(0);
    const auto dilationsVal = parseIntArrayAttr<int64_t>(dilations());
    if (outputElemType.isa<mlir::quant::UniformQuantizedPerAxisType>() && dilationsVal.size() > 2) {
        // E#84659: implement propagate type up for per channel, currently it leads to failures in later passes.
        return;
    }

    for (size_t inputInd = 0; inputInd < info.getNumInputs(); ++inputInd) {
        info.setInput(inputInd, outputElemType);
    }
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ExpandDilatedOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto dilationsVal = parseIntArrayAttr<int64_t>(dilations());
        return attr.expandDilated(ShapeRef(dilationsVal));
    }

    return nullptr;
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/propagate_quantize_dequantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceMaxOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceMaxOpAdaptor reduceMax(operands, attrs);
    if (mlir::failed(reduceMax.verify(loc))) {
        return mlir::failure();
    }
    if (reduceMax.axes() != nullptr && reduceMax.axes_value().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    } else if (reduceMax.axes() == nullptr && !reduceMax.axes_value().has_value()) {
        return errorAt(loc, "Axes was not provided properly");
    }

    const auto input = reduceMax.input();
    const auto keepDims = reduceMax.keep_dims();

    auto axesValue = IE::extractAxes(loc, reduceMax);

    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axesValue, inferredReturnShapes);
}

//
// inferElemTypeInfo
//

void vpux::IE::ReduceMaxOp::inferElemTypeInfo(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    auto arch = VPU::getArch(*this);
    if (arch == VPU::ArchKind::VPUX30XX) {
        // Workaround : Do not propagate for VPU30XX until E#80362 is implemented.
        return;
    }

    // E#84659: implement propagate type up for per channel, currently it leads to failures in later passes.
    propagateElementTypeDown(info);
}

void vpux::IE::ReduceMaxOp::inferElemTypeInfoUp(vpux::IE::LayerDataInfo<mlir::Type>& info) {
    auto arch = VPU::getArch(*this);
    if (arch == VPU::ArchKind::VPUX30XX) {
        // Workaround : Do not propagate for VPU30XX until E#80362 is implemented.
        return;
    }

    // E#84659: implement propagate type up for per channel, currently it leads to failures in later passes.
    propagateElementTypeUp(info);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReduceMaxOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType()) {
        return input();
    }

    return nullptr;
}

void vpux::IE::ReduceMaxOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr<vpux::IE::ReduceMaxOp>>(context);
}

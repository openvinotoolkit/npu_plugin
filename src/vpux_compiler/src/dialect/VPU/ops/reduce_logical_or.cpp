//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReduceLogicalOrOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::ReduceLogicalOrOpAdaptor reduceLogicalOr(operands, attrs);
    if (mlir::failed(reduceLogicalOr.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceLogicalOr.input();
    const auto keepDims = reduceLogicalOr.keep_dims() != nullptr;
    auto axes = IE::constInputToData(loc, reduceLogicalOr.axes());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    auto axesValue = axes.getValue();

    return VPU::inferReduceReturnTypes(loc, input, keepDims, axesValue, inferredReturnTypes);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ReduceLogicalOrOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::ReduceParamsBuilder builder(writer);

    EMU::BlobWriter::String type;
    type = writer.createString("logicalor");

    builder.add_keep_dims(checked_cast<bool>(keep_dims()));
    builder.add_operation(type);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReduceParams});
}

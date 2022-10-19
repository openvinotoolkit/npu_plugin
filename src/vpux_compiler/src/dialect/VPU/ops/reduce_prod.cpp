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

mlir::LogicalResult vpux::VPU::ReduceProdOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::ReduceProdOpAdaptor reduceProd(operands, attrs);
    if (mlir::failed(reduceProd.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceProd.input();
    const auto keepDims = reduceProd.keep_dims() != nullptr;
    auto axes = IE::constInputToData(loc, reduceProd.axes());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    return VPU::inferReduceReturnTypes(loc, input, keepDims, axes.getValue(), inferredReturnTypes);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ReduceProdOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::ReduceParamsBuilder builder(writer);

    EMU::BlobWriter::String type;
    type = writer.createString("prod");

    builder.add_keep_dims(checked_cast<bool>(keep_dims()));
    builder.add_operation(type);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReduceParams});
}

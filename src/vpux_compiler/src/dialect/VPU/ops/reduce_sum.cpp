//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReduceSumOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             mlir::Optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::ReduceSumOpAdaptor reduceSum(operands, attrs);
    if (mlir::failed(reduceSum.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceSum.input();
    const auto keepDims = reduceSum.keep_dims();
    auto axes = IE::constInputToData(loc, reduceSum.axes());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    auto axesValue = axes.getValue();

    return VPU::inferReduceReturnTypes(loc, input, keepDims, axesValue, inferredReturnTypes);
}

//
// inferLayoutInfo
//

void vpux::VPU::ReduceSumOp::inferLayoutInfo(mlir::Operation* op, vpux::IE::LayerLayoutInfo& info) {
    vpux::IE::inferReduceLayoutInfo(op, info);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ReduceSumOp::serialize(EMU::BlobWriter& writer) {
    EMU::BlobWriter::String type;
    type = writer.createString("sum");

    MVCNN::ReduceParamsBuilder builder(writer);
    builder.add_keep_dims(checked_cast<bool>(keep_dims()));
    builder.add_operation(type);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReduceParams});
}

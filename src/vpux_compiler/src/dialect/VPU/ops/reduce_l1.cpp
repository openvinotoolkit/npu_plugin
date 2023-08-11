//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReduceL1Op::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            mlir::Optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReduceL1OpAdaptor reduceL1(operands, attrs);
    if (mlir::failed(reduceL1.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceL1.input();
    const auto keepDims = reduceL1.keep_dims();
    auto axes = IE::constInputToData(loc, reduceL1.axes());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    auto axesValue = axes.getValue();

    return VPU::inferReduceReturnTypes(loc, input, keepDims, axesValue, inferredReturnTypes);
}

//
// inferLayoutInfo
//

void vpux::VPU::ReduceL1Op::inferLayoutInfo(mlir::Operation* op, vpux::IE::LayerLayoutInfo& info) {
    vpux::IE::inferReduceLayoutInfo(op, info);
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ReduceL1Op::serialize(EMU::BlobWriter& writer) {
    MVCNN::ReduceParamsBuilder builder(writer);

    EMU::BlobWriter::String type;
    type = writer.createString("l1");

    builder.add_keep_dims(checked_cast<bool>(keep_dims()));
    builder.add_operation(type);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReduceParams});
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ConvertOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                           mlir::Optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ConvertOpAdaptor cvt(operands, attrs);
    if (mlir::failed(cvt.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = cvt.input().getType().cast<vpux::NDTypeInterface>();
    const auto dstElemType = cvt.dstElemType();

    const auto outType = inType.changeElemType(dstElemType);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

bool vpux::VPU::ConvertOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return false;
    }

    const auto input = inputs.front().dyn_cast<vpux::NDTypeInterface>();
    const auto output = outputs.front().dyn_cast<vpux::NDTypeInterface>();

    if (!input || !output || input.getShape() != output.getShape()) {
        return false;
    }

    return true;
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ConvertOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::ConvertParamsBuilder builder(writer);
    builder.add_scale(checked_cast<float>(1.0));
    builder.add_bias(checked_cast<float>(0.0));
    builder.add_from_detection_output(false);
    builder.add_have_batch(false);
    builder.add_batch_id(0);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ConvertParams});
}

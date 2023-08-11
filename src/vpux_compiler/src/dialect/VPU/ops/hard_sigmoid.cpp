//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::HardSigmoidOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               mlir::Optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::HardSigmoidOpAdaptor hardSigmoid(operands, attrs);
    if (mlir::failed(hardSigmoid.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = hardSigmoid.input().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::HardSigmoidOp::serialize(EMU::BlobWriter& writer) {
    const auto alpha = alpha_valueAttr().getValueAsDouble();
    const auto beta = beta_valueAttr().getValueAsDouble();
    const auto sigmoid = MVCNN::CreateHardSigmoidParams(writer, checked_cast<float>(alpha), checked_cast<float>(beta));

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_HardSigmoidParams);
    builder.add_nested_params(sigmoid.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

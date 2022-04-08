//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::TanhOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::TanhOpAdaptor tanh(operands, attrs);
    if (mlir::failed(tanh.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = tanh.input().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// NCEOpInterface
//

bool vpux::VPU::TanhOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    return strategy == VPU::MultiClusterStrategy::Clustering;
}

//
// SWOpInterface
//

bool vpux::VPU::TanhOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output) {
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(
                   getArch(getOperation()), {input.getTotalAllocSize(), output.getTotalAllocSize()}) <=
           getTotalCMXSize(getOperation());
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::TanhOp::serialize(EMU::BlobWriter& writer) {
    const auto tanh = MVCNN::CreateTanhParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_TanhParams);
    builder.add_nested_params(tanh.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}

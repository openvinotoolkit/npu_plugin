//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MVNOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                       mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                       mlir::RegionRange /*regions*/,
                                                       mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::MVNOpAdaptor mvn(operands, attrs);
    if (mlir::failed(mvn.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mvn.input().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape();
    if (inShape.size() != 4 && inShape.size() != 5) {
        return errorAt(loc, "First input tensor should have 4 or 5 dimensions");
    }

    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

void vpux::VPU::MVNOp::inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
    // [Track number: E#56050]
    VPU::inferLayoutInfoSameInOutSpecificDimsOrder(
            info, {DimsOrder::NCHW, DimsOrder::NCWH, DimsOrder::NHWC, DimsOrder::NWHC});
}

//
// NCEOpInterface
//

bool vpux::VPU::MVNOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    return strategy == VPU::MultiClusterStrategy::Clustering || strategy == VPU::MultiClusterStrategy::SplitOverKernel;
}

//
// SWOpInterface
//

bool vpux::VPU::MVNOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output) {
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(
                   getArch(getOperation()), {input.getTotalAllocSize(), output.getTotalAllocSize()}) <=
           getTotalCMXSize(getOperation());
}

//
// build
//

void vpux::VPU::MVNOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                             ::mlir::BoolAttr across_channels, ::mlir::BoolAttr normalize_variance,
                             ::mlir::FloatAttr eps) {
    build(builder, state, input.getType(), input, across_channels, normalize_variance, eps, {});
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::MVNOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::MVNParamsBuilder builder(writer);
    builder.add_across_channels(across_channels());
    builder.add_normalize_variance(normalize_variance());
    builder.add_eps(static_cast<float>(eps().convertToDouble()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_MVNParams});
}

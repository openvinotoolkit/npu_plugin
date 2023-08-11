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
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

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
// ClusteredOpInterface
//

bool vpux::VPU::MVNOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }
    // MVN can only apply SOK for layer normalization
    // i.e. when the across_channel is false, the mean value is calculated over H and W
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel && !across_channels()) {
        return true;
    }
    return false;
}

//
// SWOpInterface
//

bool vpux::VPU::MVNOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2, "MVNOp requires 1 input and 1 output, but the number of buffer is {0}",
                      buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::MVNOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::MVNOp::supportCycleCostCalculation() {
    return true;
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

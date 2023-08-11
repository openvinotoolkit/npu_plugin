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
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::TanhOpAdaptor tanh(operands, attrs);
    if (mlir::failed(tanh.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = tanh.input().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::TanhOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    // Track [E#68740]
    // SOK is temporarily disabled because SEGMENTED SOK causes spilling and performance regression
    // Need to enable SOK after subgraph opt's refactoring
    return strategy == VPU::MultiClusterStrategy::Clustering || strategy == VPU::MultiClusterStrategy::SplitOverHeight;
}

//
// SWOpInterface
//

bool vpux::VPU::TanhOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2, "TanhOp requires 1 input and 1 output, but the number of buffer is {0}",
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

bool vpux::VPU::TanhOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::TanhOp::supportCycleCostCalculation() {
    return false;
}

bool vpux::VPU::TanhOp::availableSingleMerge() {
    // do not fuse post ops as first op in the block
    return false;
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

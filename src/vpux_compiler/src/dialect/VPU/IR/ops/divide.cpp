//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DivideOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DivideOpAdaptor divide(operands, attrs);
    if (mlir::failed(divide.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = divide.getInput1().getType().cast<vpux::NDTypeInterface>();
    const auto in2Type = divide.getInput2().getType().cast<vpux::NDTypeInterface>();

    const auto outShapeRes = IE::broadcastEltwiseShape(in1Type.getShape().raw(), in2Type.getShape().raw(),
                                                       divide.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        const auto outType = in1Type.changeShape(Shape(outShapeRes.value()));
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}

void vpux::VPU::DivideOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input1,
                                ::mlir::Value input2, vpux::IE::AutoBroadcastTypeAttr auto_broadcast) {
    build(odsBuilder, odsState, input1, input2, auto_broadcast.getValue(), nullptr);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::DivideOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::DivideOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::ArrayAttr /*kernel*/,
        vpux::VPU::PaddingAttr /*pad*/, mlir::ArrayAttr /*stride*/, mlir::UnitAttr uniformDistributedSegments) {
    return VPU::getSWExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                                   distributionMode, numTiles, numClusters, alignment,
                                                   uniformDistributedSegments);
}

bool VPU::DivideOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto divideOp = mlir::cast<VPU::DivideOp>(getOperation());
    const auto outputType = divideOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(divideOp, outputType.getShape()[Dims4D::Act::C], strategy);
    auto distInput1Type =
            getDistributedActivationTypeFromOp(divideOp, divideOp.getInput1().getType(), numClusters, strategy);
    auto distInput2Type =
            getDistributedActivationTypeFromOp(divideOp, divideOp.getInput2().getType(), numClusters, strategy);
    auto distOutputType =
            getDistributedOutputTypeFromOp(divideOp, divideOp.getOutput().getType(), numClusters, strategy);
    return fitIntoCMX({distInput1Type, distInput2Type, distOutputType}, reservedMem);
}

//
// SWOpInterface
//

bool vpux::VPU::DivideOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3, "DivideOp requires 2 input and 1 output, but the number of buffer is {0}",
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

bool vpux::VPU::DivideOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::DivideOp::supportCycleCostCalculation() {
    return true;
}

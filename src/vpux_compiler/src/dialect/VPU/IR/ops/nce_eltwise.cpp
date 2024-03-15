//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEEltwiseOp::fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2,
                                         vpux::NDTypeInterface output) {
    if (this->getIsInplace().value_or(false)) {
        return VPU::NCEEltwiseOp::fitIntoCMX(input1, input2, Byte(0));
    }

    return fitIntoCMX(input1, input2, output, Byte(0));
}

bool vpux::VPU::NCEEltwiseOp::fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2,
                                         vpux::NDTypeInterface output, Byte reservedMem) {
    if (this->getIsInplace().value_or(false)) {
        return VPU::NCEEltwiseOp::fitIntoCMX(input1, input2, reservedMem);
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    SmallVector<Byte> buffers = {input1.getTotalAllocSize(), input2.getTotalAllocSize(), output.getTotalAllocSize()};

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::NCEEltwiseOp::fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2, Byte reservedMem) {
    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    SmallVector<Byte> buffers = {input1.getTotalAllocSize(), input2.getTotalAllocSize()};
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

//
// isSupported
//

bool vpux::VPU::NCEEltwiseOp::isSupported(mlir::Operation* op, bool allowDifferentScales, bool allowDifferentZp,
                                          LogCb logCb, bool checkLayout, bool checkChannelAlignment) {
    if (op->getNumOperands() != 2) {
        return false;
    }
    auto input1Type = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto input2Type = op->getOperand(1).getType().cast<vpux::NDTypeInterface>();
    auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    return vpux::VPU::isNCEEltwiseSupported(getArch(op), input1Type, input2Type, outputType, allowDifferentScales,
                                            allowDifferentZp, checkLayout, checkChannelAlignment, logCb);
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEEltwiseOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEEltwiseOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto shape1 = getShape(op.getInput1());
    const auto shape2 = getShape(op.getInput2());

    if (shape1 != shape2) {
        return errorAt(loc, "Broadcasting is not supported for {0} operation", NCEEltwiseOp::getOperationName());
    }

    auto inputType = op.getInput1().getType();
    if (auto sparseInputType = inputType.dyn_cast<VPU::SparseTensorType>()) {
        inputType = sparseInputType.getData().cast<vpux::NDTypeInterface>();
    }

    inferredReturnTypes.push_back(inputType);
    return mlir::success();
}

//
// NCEOpInterface
//

SmallVector<int64_t> vpux::VPU::NCEEltwiseOp::getKernelSizeVal() {
    return {1, 1};
}

SmallVector<int64_t> vpux::VPU::NCEEltwiseOp::getStridesVal() {
    return {1, 1};
}

vpux::VPU::PaddingAttr vpux::VPU::NCEEltwiseOp::getPad() {
    return VPU::getPaddingAttr(getContext(), PadInfo(0, 0, 0, 0));
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEEltwiseOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy) {
    if (this->getIsInplace().value_or(false)) {
        return strategy == VPU::MultiClusterStrategy::Clustering ||
               strategy == VPU::MultiClusterStrategy::SplitOverHeight;
    }

    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

vpux::VPU::DistributedTensorAttr vpux::VPU::NCEEltwiseOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::ArrayAttr kernel, vpux::VPU::PaddingAttr pad,
        mlir::ArrayAttr stride, mlir::UnitAttr uniformDistributedSegments) {
    return VPU::getNCEExplicitDistributedTensorAttr(mlir::dyn_cast<VPU::NCEOpInterface>(getOperation()), shape,
                                                    distributionMode, numTiles, numClusters, alignment, kernel, pad,
                                                    stride, uniformDistributedSegments);
}

bool VPU::NCEEltwiseOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& outputTile) {
    return VPU::isOperationSplitOverHeightCompatible(getOperation(), outputTile);
}

bool VPU::NCEEltwiseOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEEltwiseOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEEltwiseOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto nceOp = mlir::cast<VPU::NCEEltwiseOp>(getOperation());
    const auto outputType = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputType.getShape()[Dims4D::Act::C], strategy);
    return fitIntoCMX(getDistributedActivationTypeFromOp(nceOp, nceOp.getInput1().getType(), numClusters, strategy),
                      getDistributedActivationTypeFromOp(nceOp, nceOp.getInput2().getType(), numClusters, strategy),
                      getDistributedOutputTypeFromOp(nceOp, nceOp.getOutput().getType(), numClusters, strategy),
                      reservedMem);
}

mlir::LogicalResult vpux::VPU::NCEEltwiseOp::verifyInputType(vpux::NDTypeInterface inputType) {
    return mlir::success(vpux::VPU::NCEInvariant::isInputActTypeSupported(VPU::getArch(*this), inputType,
                                                                          getInputChannelAlignment(), false));
}

bool vpux::VPU::NCEEltwiseOp::isVFSupported() {
    return vpux::VPU::isVFNCESupported(*this);
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEEltwiseOp::sparsitySupport() {
    const auto arch = getArch(getOperation());
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return VPU::SparsitySupport::NONE;
    case VPU::ArchKind::VPUX37XX:
        // TODO E#66913: enable input sparsity support once inputs are aligned with respect to storage element size
        return VPU::SparsitySupport::SPARSE_OUTPUTS;
    default:
        VPUX_THROW("Unknown sparsity support mode for {0}", arch);
    }
}
//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEEltwiseOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    return backInferEltwiseTile(this->getOperation(), outputTile);
}

void vpux::VPU::NCEEltwiseOp::adjustAttrs(const TilingInfo&, const TileInfo&) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEEltwiseOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

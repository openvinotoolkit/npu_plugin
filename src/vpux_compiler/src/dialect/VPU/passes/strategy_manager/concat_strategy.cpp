//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/VPU/layer_strategy.hpp"

using namespace vpux;
using namespace VPU;

SmallVector<VPU::DistributedTypeInterface> ConcatStrategy::getDistributedTensorType(
        VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy) const {
    auto origOp = mlir::dyn_cast<VPU::ConcatOp>(clusteredOp.getOperation());
    VPUX_THROW_UNLESS(origOp != nullptr, "Got non VPU::ConcatOp operation {0}", clusteredOp->getName());
    auto numClusters = VPU::getOptimalNumClusters(
            clusteredOp, origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C],
            strategy);

    SmallVector<VPU::DistributedTypeInterface> distributedTensorTypes;
    for (auto input : origOp.inputs()) {
        distributedTensorTypes.push_back(
                getDistributedActivationTypeFromOp(clusteredOp, input.getType(), numClusters, strategy));
    }
    distributedTensorTypes.push_back(
            getDistributedOutputTypeFromOp(clusteredOp, origOp->getResult(0).getType(), numClusters, strategy));

    return distributedTensorTypes;
}

bool ConcatStrategy::isOperationSplitOverHeightCompatible(VPU::ClusteredOpInterface clusteredOp,
                                                          ShapeRef outputShape) const {
    // Concat could have different H for input and output.
    // For example, there's a concat on H axis. It has two inputs with shape of 1x1x1x16 #NHWC and a output with shape
    // of 1x2x1x16 #NHWC. Then we can't assign SplitOverHeight to it because the input can't be tiled on H.
    if (auto concatOp = mlir::dyn_cast<VPU::ConcatOp>(clusteredOp.getOperation())) {
        for (auto concatInput : concatOp.inputs()) {
            auto IH = getShape(concatInput)[Dims4D::Act::H];
            if (IH < _minimumOutputHeightForSOH) {
                return false;
            }
        }
    }

    return BaseLayerStrategy::isOperationSplitOverHeightCompatible(clusteredOp, outputShape);
}

bool ConcatStrategy::isOperationSplitOverKernelCompatible(VPU::ClusteredOpInterface clusteredOp,
                                                          ShapeRef outputShape) const {
    // Channel alignment is specific for NCE DPU operations and CMX CONCAT
    auto minChannelSize = _numChannelAlignment * _numClusters;

    // Concat could have different C for input and output.
    // For example, there's a concat on C axis. It has two inputs with shape of 1x1x1x16 #NHWC and a output with shape
    // of 1x1x1x32 #NHWC. Then we can't assign SplitOverKernel to it because the input can't be tiled on C.
    if (auto concatOp = mlir::dyn_cast<VPU::ConcatOp>(clusteredOp.getOperation())) {
        for (auto concatInput : concatOp.inputs()) {
            auto IC = getShape(concatInput)[Dims4D::Act::C];
            if (IC < minChannelSize) {
                return false;
            }
        }
    }

    return BaseLayerStrategy::isOperationSplitOverKernelCompatible(clusteredOp, outputShape);
}

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <map>
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/logging.hpp"

namespace vpux {
namespace VPU {

constexpr int64_t KMB_DPU_CHANNELS_ALIGNMENT = 16;
constexpr StringLiteral multiClusterStrategy = "multiClusterStrategy";

int64_t getNumberOfClustersForSOKToAvoidAlignment(int64_t outputChannels, int64_t numClustersForCompilation);
SmallVector<int64_t> getActivationTensorNumTiles(int64_t numClustersAvailableForCompilation,
                                                 VPU::MultiClusterStrategy strategy);
Optional<SmallVector<int64_t>> getActivationTensorAlignment(VPU::NCEOpInterface nceOp,
                                                            VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getOutputTensorNumTiles(VPU::NCEOpInterface nceOp, int64_t numClustersAvailableForCompilation,
                                             VPU::MultiClusterStrategy strategy);
Optional<SmallVector<int64_t>> getOutputTensorAlignment(VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getWeightsTensorNumTiles(VPU::NCEOpInterface nceOp, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy);
Optional<SmallVector<int64_t>> getWeightsTensorAlignment(VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getWeightsTableTensorNumTiles(VPU::NCEOpInterface nceOp,
                                                   int64_t numClustersAvailableForCompilation,
                                                   VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getActivationWindowTensorNumTiles(VPU::MultiClusterStrategy strategy);
DistributionMode getActivationTensorDistributionMode(VPU::MultiClusterStrategy strategy);
DistributionMode getWeightsTensorDistributionMode(VPU::MultiClusterStrategy strategy);
DistributionMode getOutputTensorDistributionMode(VPU::MultiClusterStrategy strategy);
DistributionMode getActivationWindowTensorDistributionMode(VPU::MultiClusterStrategy strategy);
NCEClusterTilingOp createDistributedCopyOut(VPU::NCEOpInterface nceOp, NCEClusterTilingOp clusterTilingOp);
int64_t getSOHPerClusterHeightAlignment(int64_t inputWidth);
bool isSOHSupportedByDPU(ShapeRef inputShape, int64_t KY, int64_t numClusters, bool DWTypeOp);

NCEClusterTilingOp createDistributedCopyIn(VPU::NCEOpInterface nceOp, mlir::Value input,
                                           DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                           mlir::ArrayAttr alignment, VPU::MultiClusterStrategy strategy);

DistributedTensorType createDistributedTensorType(VPU::NCEOpInterface nceOp, mlir::Value input,
                                                  DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                  mlir::ArrayAttr alignment, VPU::MultiClusterStrategy strategy);

}  // namespace VPU
}  // namespace vpux

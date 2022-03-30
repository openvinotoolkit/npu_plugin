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
constexpr StringLiteral splitOverHeightOverlapped =
        "SplitOverHeightOverlapped";  // Strategy is for channel major convolution
constexpr StringLiteral splitOverHeight = "SplitOverHeight";
constexpr StringLiteral splitOverKernel = "SplitOverKernel";
constexpr StringLiteral clustering = "Clustering";

int64_t getNumberOfClustersForSOKToAvoidAlignment(int64_t outputChannels, int64_t numClustersForCompilation);
SmallVector<int64_t> getActivationTensorNumTiles(int64_t numClustersAvailableForCompilation, StringRef strategy);
Optional<SmallVector<int64_t>> getActivationTensorAlignment(VPU::NCEOpInterface nceOp, StringRef strategy);
SmallVector<int64_t> getOutputTensorNumTiles(VPU::NCEOpInterface nceOp, int64_t numClustersAvailableForCompilation,
                                             StringRef strategy);
Optional<SmallVector<int64_t>> getOutputTensorAlignment(StringRef strategy);
SmallVector<int64_t> getWeightsTensorNumTiles(VPU::NCEOpInterface nceOp, int64_t numClustersAvailableForCompilation,
                                              StringRef strategy);
Optional<SmallVector<int64_t>> getWeightsTensorAlignment(StringRef strategy);
SmallVector<int64_t> getWeightsTableTensorNumTiles(VPU::NCEOpInterface nceOp,
                                                   int64_t numClustersAvailableForCompilation, StringRef strategy);
SmallVector<int64_t> getActivationWindowTensorNumTiles(StringRef strategy);
DistributionMode getActivationTensorDistributionMode(StringRef strategy);
DistributionMode getWeightsTensorDistributionMode(StringRef strategy);
DistributionMode getOutputTensorDistributionMode(StringRef strategy);
DistributionMode getActivationWindowTensorDistributionMode(StringRef strategy);
NCEClusterTilingOp createDistributedCopyOut(VPU::NCEOpInterface nceOp, NCEClusterTilingOp clusterTilingOp);
int64_t getSOHPerClusterHeightAlignment(int64_t inputWidth);
bool isSOHSupportedByDPU(ShapeRef inputShape, int64_t KY, int64_t numClusters, bool DWTypeOp);

NCEClusterTilingOp createDistributedCopyIn(VPU::NCEOpInterface nceOp, mlir::Value input,
                                           DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                           mlir::ArrayAttr alignment, StringRef strategy);

DistributedTensorType createDistributedTensorType(VPU::NCEOpInterface nceOp, mlir::Value input,
                                                  DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                  mlir::ArrayAttr alignment, StringRef strategy);

}  // namespace VPU
}  // namespace vpux

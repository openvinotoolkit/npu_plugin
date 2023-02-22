//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <set>
#include <unordered_map>
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
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
const SmallVector<int64_t> DISTRIBUTED_C_ALIGNMENT = SmallVector<int64_t>{1, 16, 1, 1};

bool isSegmentedSWOp(mlir::Operation* op);
bool inputProducersCompatible(mlir::Operation* op);
bool isSegmentedInputCompatible(mlir::Operation* op);
bool isSegmentedOutputCompatible(mlir::Operation* op);
int64_t getNumberOfClustersForSOKToAvoidAlignment(int64_t outputChannels, int64_t numClustersForCompilation);
int64_t getNumberOfClustersForSOH(int64_t outputHeight, int64_t numClustersForCompilation);
SmallVector<int64_t> getActivationTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                 int64_t numClustersAvailableForCompilation,
                                                 VPU::MultiClusterStrategy strategy);
Optional<SmallVector<int64_t>> getActivationTensorAlignment(VPU::ClusteredOpInterface clusteredOp,
                                                            mlir::IntegerAttr numClusters,
                                                            VPU::MultiClusterStrategy strategy,
                                                            vpux::NDTypeInterface inputType = nullptr);
SmallVector<int64_t> getOutputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                             int64_t numClustersAvailableForCompilation,
                                             VPU::MultiClusterStrategy strategy);
Optional<SmallVector<int64_t>> getOutputTensorAlignment(VPU::MultiClusterStrategy strategy);
Optional<vpux::NDTypeInterface> adjustOutputAlignmentForSOH(VPU::ClusteredOpInterface clusteredOp,
                                                            vpux::NDTypeInterface originalDistType);

SmallVector<int64_t> getWeightsTensorNumTiles(vpux::NDTypeInterface tensorType,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy);
Optional<SmallVector<int64_t>> getWeightsTensorAlignment(VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getWeightsTableTensorNumTiles(vpux::NDTypeInterface tensorType,
                                                   int64_t numClustersAvailableForCompilation,
                                                   VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getInstructionListTableTensorNumTiles(VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getActivationWindowTensorNumTiles(VPU::MultiClusterStrategy strategy);
DistributionMode getActivationTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                     VPU::MultiClusterStrategy strategy);
DistributionMode getWeightsTensorDistributionMode(VPU::MultiClusterStrategy strategy);
DistributionMode getInstructionListTableTensorDistributionMode(VPU::MultiClusterStrategy strategy);
DistributionMode getOutputTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                 VPU::MultiClusterStrategy strategy);
DistributionMode getActivationWindowTensorDistributionMode(VPU::MultiClusterStrategy strategy);
NCEClusterTilingOp createDistributedCopyOut(VPU::ClusteredOpInterface clusteredOp, NCEClusterTilingOp clusterTilingOp);
NCEClusterTilingOp createDistributedCopyOut(mlir::Operation* sourceOp, vpux::NDTypeInterface outputType);
int64_t getSOHPerClusterHeightAlignment(int64_t inputWidth);
int64_t getSOHMinimalHeightAlignment(vpux::ShapeRef shape, int64_t numClusters);
bool isSOHSupportedByDPU(ShapeRef inputShape, int64_t numClusters, bool DWTypeOp, VPU::ArchKind arch);

mlir::IntegerAttr getOptimalNumClusters(VPU::ClusteredOpInterface clusteredOp, int64_t OC,
                                        VPU::MultiClusterStrategy strategy);
NCEClusterTilingOp createDistributedCopyIn(VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
                                           DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                           mlir::ArrayAttr alignment, VPU::MultiClusterStrategy strategy);

VPU::DistributedTensorType createDistributedTensorType(VPU::ClusteredOpInterface clusteredOp,
                                                       vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                       mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment);

VPU::DistributedTensorType createDistributedTensorType(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                       mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment);

VPU::DistributedTensorType createDistributedTensorType(VPU::SWOpInterface swOp, vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                       mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment);

VPU::DistributedTypeInterface getDistributedActivationTypeFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                                 vpux::NDTypeInterface inputType,
                                                                 mlir::IntegerAttr numClusters);
VPU::DistributedTypeInterface getDistributedFilterTypeFromOp(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                             mlir::IntegerAttr numClusters);

VPU::DistributedTypeInterface getDistributedOutputTypeFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                             vpux::NDTypeInterface inputType,
                                                             mlir::IntegerAttr numClusters);

VPU::DistributedTypeInterface getDistributedActivationTypeFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                                 vpux::NDTypeInterface inputType,
                                                                 mlir::IntegerAttr numClusters,
                                                                 VPU::MultiClusterStrategy customStrategy);
VPU::DistributedTypeInterface getDistributedFilterTypeFromOp(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                             mlir::IntegerAttr numClusters,
                                                             VPU::MultiClusterStrategy customStrategy);
VPU::DistributedTypeInterface getDistributedOutputTypeFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                             vpux::NDTypeInterface inputType,
                                                             mlir::IntegerAttr numClusters,
                                                             VPU::MultiClusterStrategy customStrategy);

Shape getLargestClusterOutputShape(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy);
SmallVector<Shape> getPerClusterOutputShape(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy);
bool archRequiresDWOpsHeightAlign(ArchKind arch);
bool clusteredSWOpHasAlignedInput(VPU::ClusteredOpInterface swOp);
bool clusteredSWOpHasAlignedOutput(VPU::ClusteredOpInterface swOp);
bool isDWOpAndNeedsAlign(ArchKind arch, VPUIP::NCETaskType nceTaskType);
bool isEltwiseOpAndNeedsAlign(VPU::ClusteredOpInterface nceOp);
bool swOpInputNeedsAlign(VPU::ClusteredOpInterface swOp);

}  // namespace VPU
}  // namespace vpux

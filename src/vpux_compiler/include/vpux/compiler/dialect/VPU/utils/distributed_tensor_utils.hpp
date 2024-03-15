//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <set>
#include <unordered_map>
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace VPU {

constexpr int64_t KMB_DPU_CHANNELS_ALIGNMENT = 16;
constexpr StringLiteral multiClusterStrategy = "multiClusterStrategy";
const SmallVector<int64_t> DISTRIBUTED_C_ALIGNMENT = SmallVector<int64_t>{1, 16, 1, 1};

bool isSegmentedSWOp(mlir::Operation* op);
bool inputProducersCompatible(mlir::Operation* op, mlir::DenseSet<mlir::Operation*> handledUsers = {});
bool isSegmentedInputCompatible(mlir::Operation* op, mlir::DenseSet<mlir::Operation*> handledUsers = {});
bool isSegmentedOutputCompatible(mlir::Operation* op);
int64_t getNumberOfClustersForSOKToAvoidAlignment(int64_t outputChannels, int64_t numClustersForCompilation,
                                                  bool uniformDistributedSegments = true);
int64_t getNumberOfClustersForSpatialDim(int64_t outputSpatialDim, int64_t numClustersForCompilation,
                                         bool uniformDistributedSegments = true);
SmallVector<int64_t> getActivationTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                 int64_t numClustersAvailableForCompilation,
                                                 VPU::MultiClusterStrategy strategy);
std::optional<SmallVector<int64_t>> getActivationTensorAlignment(VPU::ClusteredOpInterface clusteredOp,
                                                                 mlir::IntegerAttr numClusters,
                                                                 VPU::MultiClusterStrategy strategy,
                                                                 vpux::NDTypeInterface inputType = nullptr,
                                                                 vpux::NDTypeInterface outputType = nullptr);
SmallVector<int64_t> getOutputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                             int64_t numClustersAvailableForCompilation,
                                             VPU::MultiClusterStrategy strategy);
std::optional<SmallVector<int64_t>> getOutputTensorAlignment(VPU::MultiClusterStrategy strategy);
std::optional<vpux::NDTypeInterface> adjustOutputAlignmentForSOH(VPU::ClusteredOpInterface clusteredOp,
                                                                 vpux::NDTypeInterface originalDistType);

SmallVector<int64_t> getWeightsTensorNumTiles(VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface tensorType,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy);
std::optional<SmallVector<int64_t>> getWeightsTensorAlignment(VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getWeightsTableTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                   vpux::NDTypeInterface tensorType,
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
int64_t getSOHPerClusterHeightAlignment(int64_t inputWidth, bool isInputSparse);
int64_t getSOHMinimalHeightAlignment(vpux::ShapeRef shape, int64_t numClusters, bool isInputSparse, VPU::ArchKind arch);
bool isSOHSupportedByDPU(vpux::NDTypeInterface inputType, ShapeRef inputShape, int64_t numClusters, bool DWTypeOp,
                         VPU::ArchKind arch);

NCEClusterTilingOp createDistributedCopyIn(VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
                                           DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                           mlir::ArrayAttr alignment, VPU::MultiClusterStrategy strategy,
                                           const bool hasExplicitDistributedAttr);

VPU::DistributedTensorType createExplicitDistributedTensorType(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, DistributionMode distributionMode,
        mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment,
        mlir::ArrayAttr kernel = nullptr, VPU::PaddingAttr pad = nullptr, mlir::ArrayAttr stride = nullptr);

VPU::DistributedTensorType createDistributedTensorType(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, DistributionMode distributionMode,
        mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment,
        const bool hasExplicitDistributedAttr, mlir::ArrayAttr kernel = nullptr, VPU::PaddingAttr pad = nullptr,
        mlir::ArrayAttr stride = nullptr, mlir::UnitAttr equalComputeAndMemoryView = nullptr);

VPU::DistributedTensorType createDistributedTensorType(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                       mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment,
                                                       mlir::ArrayAttr kernel = nullptr, VPU::PaddingAttr pad = nullptr,
                                                       mlir::ArrayAttr stride = nullptr,
                                                       mlir::UnitAttr equalComputeAndMemoryView = nullptr);

VPU::SparseTensorType createSparseTensorDistributedType(
        VPU::ClusteredOpInterface clusteredOp, VPU::SparseTensorType sparseInputType, DistributionMode distributionMode,
        mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment,
        const bool hasExplicitDistributedAttr, mlir::ArrayAttr kernelAttr = nullptr, VPU::PaddingAttr padAttr = nullptr,
        mlir::ArrayAttr strideAttr = nullptr);

DistributedTensorType createDistributedTensorType(VPU::ConcatOp viewLikeOp, vpux::NDTypeInterface inputType,
                                                  DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                  mlir::IntegerAttr optimalNumberOfClusters, mlir::ArrayAttr alignment,
                                                  mlir::ArrayAttr kernel = nullptr, VPU::PaddingAttr pad = nullptr,
                                                  mlir::ArrayAttr stride = nullptr);

VPU::DistributedTensorType createDistributedTensorType(VPU::SWOpInterface swOp, vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                       mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment);

VPU::DistributedTypeInterface getDistributedActivationTypeFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                                 vpux::NDTypeInterface inputType,
                                                                 mlir::IntegerAttr numClusters,
                                                                 vpux::NDTypeInterface outputType = nullptr);
VPU::DistributedTypeInterface getDistributedFilterTypeFromOp(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                             mlir::IntegerAttr numClusters);

VPU::DistributedTypeInterface getDistributedOutputTypeFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                             vpux::NDTypeInterface outputType,
                                                             mlir::IntegerAttr numClusters,
                                                             vpux::NDTypeInterface inputType = nullptr,
                                                             const bool hasExplicitDistributedAttr = false);

VPU::DistributedTypeInterface getDistributedActivationTypeFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                                 vpux::NDTypeInterface inputType,
                                                                 mlir::IntegerAttr numClusters,
                                                                 VPU::MultiClusterStrategy customStrategy,
                                                                 mlir::ArrayAttr customAlignment = nullptr,
                                                                 vpux::NDTypeInterface outputType = nullptr);
VPU::DistributedTypeInterface getDistributedFilterTypeFromOp(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                             mlir::IntegerAttr numClusters,
                                                             VPU::MultiClusterStrategy customStrategy);
VPU::DistributedTypeInterface getDistributedOutputTypeFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                             vpux::NDTypeInterface outputType,
                                                             mlir::IntegerAttr numClusters,
                                                             VPU::MultiClusterStrategy customStrategy,
                                                             vpux::NDTypeInterface inputType = nullptr,
                                                             const bool hasExplicitDistributedAttr = false);

vpux::NDTypeInterface getDistributedOutputTensorType(VPU::ClusteredOpInterface clusteredOp,
                                                     mlir::IntegerAttr numClusters, VPU::MultiClusterStrategy strategy,
                                                     vpux::NDTypeInterface outputTensorType,
                                                     const bool hasExplicitDistributedAttr, bool alignForSOH = true);

bool isSegmentedOverlappedAxisSameAsSliceAxis(mlir::ArrayAttr numTiles, ArrayRef<int64_t> inputShape,
                                              ArrayRef<int64_t> sliceShape);

Shape getLargestClusterOutputShape(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy);
bool isDWOpAndNeedsAlign(ArchKind arch, VPUIP::NCETaskType nceTaskType);
bool isEltwiseOpAndNeedsAlign(VPU::ClusteredOpInterface nceOp);
bool isSWOpChannelAlignmentCompatible(VPU::ClusteredOpInterface swOp, vpux::NDTypeInterface inputType,
                                      vpux::NDTypeInterface outputType);
bool isSWOpWithAlignedInputChannelReq(VPU::ClusteredOpInterface swOp, vpux::NDTypeInterface inputType = nullptr,
                                      vpux::NDTypeInterface outputType = nullptr);
bool isSWOpWithAlignedOutputChannelReq(VPU::ClusteredOpInterface swOp, vpux::NDTypeInterface inputType = nullptr,
                                       vpux::NDTypeInterface outputType = nullptr);

VPU::DistributedTensorType composeDistributedType(VPU::ClusteredOpInterface permuteOp,
                                                  const VPU::DistributedTensorType distType,
                                                  const vpux::NDTypeInterface ndType, const mlir::ArrayAttr tileOverDim,
                                                  const mlir::ArrayAttr fusedKernel, const mlir::ArrayAttr fusedStrides,
                                                  const PaddingAttr fusedPads,
                                                  bool enableExplicitDistributedTensorAttr = false,
                                                  const mlir::UnitAttr equalComputeAndMemoryView = nullptr);
mlir::Operation* getNextCompressConv(mlir::Operation* nceOp);
mlir::Type fuseOverlapParams(VPU::ClusteredOpInterface permuteOp, const VPU::DistributedTensorType distType,
                             mlir::Operation* nextConv, bool enableExplicitDistributedTensorAttr = false);

template <typename T, std::enable_if_t<std::is_same<VPU::NCEClusterTilingOp, T>::value, bool> = true>
mlir::Value getDistributedOperandFromNCEClusterTiling(T clusterOp, mlir::Value innerOperand) {
    if (innerOperand == nullptr) {
        return nullptr;
    }
    auto blockArg = innerOperand.dyn_cast<mlir::BlockArgument>();
    if (blockArg == nullptr) {
        return nullptr;
    }
    auto operandNum = blockArg.getArgNumber();
    VPUX_THROW_UNLESS(operandNum < clusterOp.getNumOperands(),
                      "Argument number '{0}' is out of range of operands for NCEClusterTiling op '{1}'", operandNum,
                      clusterOp.getNumOperands());
    return clusterOp.getOperand(operandNum);
}

}  // namespace VPU
}  // namespace vpux

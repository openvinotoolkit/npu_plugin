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
Optional<SmallVector<int64_t>> getActivationTensorAlignment(mlir::Operation* op, StringRef strategy);
SmallVector<int64_t> getOutputTensorNumTiles(mlir::Operation* op, int64_t numClustersAvailableForCompilation,
                                             StringRef strategy);
Optional<SmallVector<int64_t>> getOutputTensorAlignment(StringRef strategy);
SmallVector<int64_t> getWeightsTensorNumTiles(mlir::Operation* op, int64_t numClustersAvailableForCompilation,
                                              StringRef strategy);
Optional<SmallVector<int64_t>> getWeightsTensorAlignment(StringRef strategy);
SmallVector<int64_t> getWeightsTableTensorNumTiles(mlir::Operation* op, int64_t numClustersAvailableForCompilation,
                                                   StringRef strategy);
SmallVector<int64_t> getActivationWindowTensorNumTiles(StringRef strategy);
DistributionMode getActivationTensorDistributionMode(StringRef strategy);
DistributionMode getWeightsTensorDistributionMode(StringRef strategy);
DistributionMode getOutputTensorDistributionMode(StringRef strategy);
DistributionMode getActivationWindowTensorDistributionMode(StringRef strategy);
NCEClusterTilingOp createDistributedCopyOut(mlir::Operation* origOp, NCEClusterTilingOp clusterTilingOp);
mlir::ArrayAttr getKernelSize(mlir::Operation* origOp);
int64_t getSOHPerClusterHeightAlignment(int64_t inputWidth);
bool isSOHSupportedByDPU(ShapeRef inputShape, int64_t KY, int64_t numClusters, bool DWTypeOp);

template <class ConcreteOp>
mlir::ArrayAttr getStride(ConcreteOp origOp) {
    return origOp.strides();
}

template <>
inline mlir::ArrayAttr getStride<NCEEltwiseOp>(NCEEltwiseOp) {
    return nullptr;
}

template <class ConcreteOp>
PaddingAttr getPad(ConcreteOp origOp) {
    return origOp.padAttr();
}

template <>
inline PaddingAttr getPad<NCEEltwiseOp>(NCEEltwiseOp) {
    return nullptr;
}

template <class ConcreteOp>
NCEClusterTilingOp createDistributedCopyIn(ConcreteOp origOp, mlir::Value input, DistributionMode distributionMode,
                                           mlir::ArrayAttr numTiles, mlir::ArrayAttr alignment, StringRef strategy) {
    auto inputTensorDistributedTensorType =
            createDistributedTensorType(origOp, input, distributionMode, numTiles, alignment, strategy);

    mlir::OpBuilder builder(origOp);
    builder.setInsertionPoint(origOp);
    const auto inputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::ValueRange newOperands) {
        const auto memSpace = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(MemoryKind::CMX_NN));
        auto inputTensorDistributedCopyOp = builder.create<IE::CopyOp>(origOp->getLoc(), newOperands[0], memSpace);
        builder.create<YieldOp>(loc, inputTensorDistributedCopyOp->getResults());
    };

    auto distributedInputCopyOp = builder.create<NCEClusterTilingOp>(origOp->getLoc(), inputTensorDistributedTensorType,
                                                                     input, inputTensorBodyBuilder);

    return distributedInputCopyOp;
}

template <class ConcreteOp>
DistributedTensorType createDistributedTensorType(ConcreteOp origOp, mlir::Value input,
                                                  DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                  mlir::ArrayAttr alignment, StringRef strategy) {
    DistributedTensorAttr distributedActivationTensorAttr;
    auto module = origOp->template getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    const auto numClustersAvailableForCompilation = getIntAttr(origOp.getContext(), nceOp.count());
    const auto activationTensorDistributionModeAttr = DistributionModeAttr::get(origOp.getContext(), distributionMode);
    mlir::IntegerAttr optimalNumberOfClusters = numClustersAvailableForCompilation;

    // Here the number of clusters to be used for an individual SOK layer is determined
    // such that additional alignment of the per cluster output channels is not required.
    // For example 80 output channels, the weights should only be split on 3 clusters [32, 32, 16].
    // Also when creating the copy-in for the activation we need to ensure that the number
    // of clusters that the input is duplicated to is also 3 clusters in this case.
    // Therefore we use the variable optimalNumberOfClusters for both purposes here, to detemine
    // num_tiles and numClusters for the activations and the weights.
    if (strategy == splitOverKernel) {
        int64_t numClustersToUseForLayer = numClustersAvailableForCompilation.getValue().getSExtValue();
        auto OC = getShape(origOp->getResult(0))[Dims4D::Act::C];
        numClustersToUseForLayer = getNumberOfClustersForSOKToAvoidAlignment(OC, numClustersToUseForLayer);
        optimalNumberOfClusters = mlir::IntegerAttr::get(getInt64Type(origOp->getContext()), numClustersToUseForLayer);
    }

    auto kernel = getKernelSize(origOp);
    const auto shape = getShape(input);
    if (distributionMode == DistributionMode::OVERLAPPED) {
        auto stride = getStride(origOp);
        auto pad = getPad(origOp);
        optimalNumberOfClusters = getIntAttr(origOp.getContext(), nceOp.count());

        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, kernel, pad, stride,
                                           optimalNumberOfClusters, alignment, origOp.getContext());
    } else if (distributionMode == DistributionMode::DUPLICATED) {
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, nullptr, nullptr, nullptr, nullptr,
                                           optimalNumberOfClusters, alignment, origOp.getContext());
    } else if (VPU ::bitEnumContains(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, nullptr, nullptr, nullptr,
                                           optimalNumberOfClusters, alignment, origOp.getContext());

    } else {
        VPUX_THROW("Unsupported distribution mode: {0}", VPU::stringifyDistributionMode(distributionMode));
    }

    const auto memSpace = vpux::IndexedSymbolAttr::get(MemoryKindAttr::get(origOp.getContext(), MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(input).toAffineMap(origOp.getContext()));
    auto elemType = input.getType().template cast<vpux::NDTypeInterface>().getElementType();

    return DistributedTensorType::get(origOp.getContext(), shape.raw(), elemType, order, memSpace,
                                      distributedActivationTensorAttr);
}

}  // namespace VPU
}  // namespace vpux

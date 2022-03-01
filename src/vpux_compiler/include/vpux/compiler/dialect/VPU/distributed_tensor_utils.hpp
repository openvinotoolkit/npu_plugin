//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <map>
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/logging.hpp"

namespace vpux {
namespace VPU {

constexpr StringLiteral multiClusterStrategy = "multiClusterStrategy";
constexpr StringLiteral splitOverHeightOverlapped =
        "SplitOverHeightOverlapped";  // Strategy is for channel major convolution
constexpr StringLiteral splitOverHeight = "SplitOverHeight";
constexpr StringLiteral splitOverKernel = "SplitOverKernel";
constexpr StringLiteral clustering = "Clustering";

SmallVector<int64_t> getActivationTensorNumTiles(mlir::Operation* op, int64_t numClusters, StringRef strategy);
SmallVector<int64_t> getOutputTensorNumTiles(int64_t numClusters, StringRef strategy);
SmallVector<int64_t> getWeightsTensorNumTiles(int64_t numClusters, StringRef strategy);
SmallVector<int64_t> getWeightsTableTensorNumTiles(int64_t numClusters, StringRef strategy);
SmallVector<int64_t> getActivationWindowTensorNumTiles(int64_t numClusters, StringRef strategy, ArchKind arch);
DistributionMode getActivationTensorDistributionMode(StringRef strategy);
DistributionMode getWeightsTensorDistributionMode(StringRef strategy);
DistributionMode getOutputTensorDistributionMode(StringRef strategy);
DistributionMode getActivationWindowTensorDistributionMode(StringRef strategy, ArchKind arch);
NCEClusterTilingOp createDistributedCopyOut(mlir::Operation* origOp, NCEClusterTilingOp clusterTilingOp);
mlir::ArrayAttr getKernelSize(mlir::Operation* origOp);

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
                                           mlir::ArrayAttr numTiles) {
    auto inputTensorDistributedTensorType = createDistributedTensorType(origOp, input, distributionMode, numTiles);

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

NCEClusterTilingOp createDistributedSpilledCopy(mlir::Operation* origOp, NCEClusterTilingOp clusterTilingOp) {
    mlir::OpBuilder builder(origOp);
    auto origOutput = origOp->getResult(0);
    const auto origOutType = origOutput.getType().cast<NDTypeInterface>();
    const auto origOutMemSpace = origOutType.getMemSpace();

    const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                             mlir::ValueRange newOperands) {
        auto outputTensorDistributedCopyOp = builder.create<IE::CopyOp>(loc, newOperands[0], origOutMemSpace);
        builder.create<YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
    };

    return builder.create<NCEClusterTilingOp>(clusterTilingOp->getLoc(), origOutType, clusterTilingOp->getResult(0),
                                              outputTensorBodyBuilder);
}

template <class ConcreteOp>
DistributedTensorType createDistributedTensorType(ConcreteOp origOp, mlir::Value input,
                                                  DistributionMode distributionMode, mlir::ArrayAttr numTiles) {
    DistributedTensorAttr distributedActivationTensorAttr;
    auto module = origOp->template getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    const auto numClusters = getIntAttr(origOp.getContext(), nceOp.count());
    const auto activationTensorDistributionModeAttr = DistributionModeAttr::get(origOp.getContext(), distributionMode);

    if (distributionMode == DistributionMode::OVERLAPPED) {
        auto kernel = getKernelSize(origOp);
        auto stride = getStride(origOp);
        auto pad = getPad(origOp);

        distributedActivationTensorAttr = DistributedTensorAttr::get(
                activationTensorDistributionModeAttr, numTiles, kernel, pad, stride, numClusters, origOp.getContext());
    } else if (distributionMode == DistributionMode::DUPLICATED) {
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, nullptr, nullptr, nullptr, nullptr,
                                           numClusters, origOp.getContext());
    } else {
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, nullptr, nullptr, nullptr,
                                           numClusters, origOp.getContext());
    }

    const auto shape = getShape(input);
    const auto memSpace = vpux::IndexedSymbolAttr::get(MemoryKindAttr::get(origOp.getContext(), MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(input).toAffineMap(origOp.getContext()));
    auto elemType = input.getType().template cast<vpux::NDTypeInterface>().getElementType();

    return DistributedTensorType::get(origOp.getContext(), shape.raw(), elemType, order, memSpace,
                                      distributedActivationTensorAttr);
}

DistributionMode getOutputTensorDistributionMode(StringRef strategy) {
    if (strategy == splitOverHeightOverlapped) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == splitOverHeight) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == splitOverKernel) {
        return DistributionMode::SEGMENTED | DistributionMode::DUPLICATED;
    } else if (strategy == clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "output tensor",
                   strategy);
    }
}

DistributionMode getActivationWindowTensorDistributionMode(StringRef strategy, ArchKind arch) {
    if (strategy == splitOverHeightOverlapped) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == splitOverHeight) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == splitOverKernel) {
        if (arch == ArchKind::MTL) {
            return DistributionMode::SEGMENTED;
        } else if (arch == ArchKind::KMB) {
            return DistributionMode::DUPLICATED;
        } else {
            VPUX_THROW("Unsupported arch {0}", arch);
        }
    } else if (strategy == clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   " activation window tensor",
                   strategy);
    }
}

}  // namespace VPU
}  // namespace vpux

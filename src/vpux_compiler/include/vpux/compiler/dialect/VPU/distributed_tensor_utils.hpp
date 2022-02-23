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
<<<<<<< HEAD
NCEClusterTilingOp createDistributedCopyIn(ConcreteOp origOp, mlir::Value input, DistributionMode distributionMode,
                                           mlir::ArrayAttr numTiles) {
    auto inputTensorDistributedTensorType = createDistributedTensorType(origOp, input, distributionMode, numTiles);
=======
NCEClusterTilingOp createDistributedInputTensor(ConcreteOp origOp, mlir::Value input, DistributionMode distributionMode,
                                                mlir::ArrayAttr numTiles) {
    auto inputTensorDistributedTensorType = createDistributedInputTensorType(origOp, input, distributionMode, numTiles);
>>>>>>> simplified code for WW10

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
<<<<<<< HEAD
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

        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, kernel, pad, stride,
                                           numClusters, nullptr, origOp.getContext());
    } else if (distributionMode == DistributionMode::DUPLICATED) {
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, nullptr, nullptr, nullptr, nullptr,
                                           numClusters, nullptr, origOp.getContext());
    } else {
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, nullptr, nullptr, nullptr,
                                           numClusters, nullptr, origOp.getContext());
    }

    const auto shape = getShape(input);
    const auto memSpace = vpux::IndexedSymbolAttr::get(MemoryKindAttr::get(origOp.getContext(), MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(input).toAffineMap(origOp.getContext()));
    auto elemType = input.getType().template cast<vpux::NDTypeInterface>().getElementType();

    return DistributedTensorType::get(origOp.getContext(), shape.raw(), elemType, order, memSpace,
                                      distributedActivationTensorAttr);
=======
DistributedTensorType createDistributedInputTensorType(ConcreteOp origOp, mlir::Value input,
                                                       DistributionMode distributionMode, mlir::ArrayAttr numTiles) {
    const auto inputTensorDistributionModeAttr = DistributionModeAttr::get(origOp.getContext(), distributionMode);

    auto kernel = getKernelSize(origOp);
    auto stride = getStride(origOp);
    auto pad = getPad(origOp);
    auto module = origOp->template getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);

    const auto numClusters = getIntAttr(origOp.getContext(), nceOp.count());

    auto inputTensorDistributedTensorAttr = DistributedTensorAttr::get(
            inputTensorDistributionModeAttr, numTiles, kernel, pad, stride, numClusters, origOp.getContext());

    const auto inputShape = getShape(input);
    const auto memSpace = vpux::IndexedSymbolAttr::get(MemoryKindAttr::get(origOp.getContext(), MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(input).toAffineMap(origOp.getContext()));
    auto elemType = input.getType().template cast<vpux::NDTypeInterface>().getElementType();

    const auto inputTensorDistributedTensorType = DistributedTensorType::get(
            origOp.getContext(), inputShape.raw(), elemType, order, memSpace, inputTensorDistributedTensorAttr);

    return inputTensorDistributedTensorType;
}

template <class ConcreteOp>
DistributedTensorType createDistributedOutputTensorType(ConcreteOp origOp, DistributionMode distributionMode,
                                                        mlir::ArrayAttr numTiles) {
    const auto outputTensorDistributionModeAttr = DistributionModeAttr::get(origOp.getContext(), distributionMode);

    auto kernel = getKernelSize(origOp);
    auto stride = getStride(origOp);
    auto pad = getPad(origOp);

    auto module = origOp->template getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);

    const auto numClusters = getIntAttr(origOp.getContext(), nceOp.count());

    auto outputTensorDistributedTensorAttr = DistributedTensorAttr::get(
            outputTensorDistributionModeAttr, numTiles, kernel, pad, stride, numClusters, origOp.getContext());

    const auto outputShape = getShape(origOp.output());
    const auto memSpace = vpux::IndexedSymbolAttr::get(MemoryKindAttr::get(origOp.getContext(), MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(origOp.output()).toAffineMap(origOp.getContext()));
    auto elemType = origOp.output().getType().template cast<vpux::NDTypeInterface>().getElementType();

    return DistributedTensorType::get(origOp.getContext(), outputShape.raw(), elemType, order, memSpace,
                                      outputTensorDistributedTensorAttr);
}

template <class ConcreteOp>
mlir::ArrayAttr getActivationTensorNumTiles(ConcreteOp origOp) {
    const StringRef strategy =
            origOp->template getAttr(multiClusterStrategy).template cast<mlir::StringAttr>().getValue();
    auto module = origOp->template getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    const auto numClusters = nceOp.count();

    if (strategy == splitOverHeightOverlapped) {
        return getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, static_cast<int>(numClusters), 1}));
    } else if (strategy == splitOverHeight) {
        return getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, static_cast<int>(numClusters), 1}));
    } else if (strategy == splitOverKernel) {
        return getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 1, 1}));
    } else if (strategy == clustering) {
        return getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 1, 1}));
    } else {
        VPUX_THROW("Operation {0} was not assigned a valid multi-cluster strategy, unable to determine the number of "
                   "tiles "
                   "for the activation tensor",
                   origOp->getName());
    }
}

template <class ConcreteOp>
mlir::ArrayAttr getWeightsTensorNumTiles(ConcreteOp origOp) {
    const StringRef strategy =
            origOp->template getAttr(multiClusterStrategy).template cast<mlir::StringAttr>().getValue();
    auto module = origOp->template getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    const auto numClusters = nceOp.count();
    if (strategy == splitOverHeightOverlapped) {
        return getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 1, 1}));
    } else if (strategy == splitOverHeight) {
        return getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 1, 1}));
    } else if (strategy == splitOverKernel) {
        return getIntArrayAttr(origOp.getContext(), makeArrayRef({static_cast<int>(numClusters), 1, 1, 1}));
    } else if (strategy == clustering) {
        return getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, 1, 1}));
    } else {
        VPUX_THROW("Operation {0} was not assigned a valid multi-cluster strategy, unable to determine the number of "
                   "tiles "
                   "for the weights tensor",
                   origOp->getName());
    }
}

template <class ConcreteOp>
DistributionMode getActivationTensorDistributionMode(ConcreteOp origOp) {
    const StringRef strategy =
            origOp->template getAttr(multiClusterStrategy).template cast<mlir::StringAttr>().getValue();
    if (strategy == splitOverHeightOverlapped) {
        return DistributionMode::OVERLAPPED;
    } else if (strategy == splitOverHeight) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == splitOverKernel) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW(
                "Operation {0} was not assigned a valid multi-cluster strategy, unable to determine the distribution "
                "mode "
                "for the activation tensor",
                origOp->getName());
    }
}

template <class ConcreteOp>
DistributionMode getWeightsTensorDistributionMode(ConcreteOp origOp) {
    const StringRef strategy =
            origOp->template getAttr(multiClusterStrategy).template cast<mlir::StringAttr>().getValue();
    if (strategy == splitOverHeightOverlapped) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == splitOverHeight) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == splitOverKernel) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW(
                "Operation {0} was not assigned a valid multi-cluster strategy, unable to determine the distribution "
                "mode "
                "for the weights tensor",
                origOp->getName());
    }
>>>>>>> simplified code for WW10
}

template <class ConcreteOp>
DistributionMode getOutputTensorDistributionMode(ConcreteOp origOp) {
    const StringRef strategy =
            origOp->template getAttr(multiClusterStrategy).template cast<mlir::StringAttr>().getValue();
    if (strategy == splitOverHeightOverlapped) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == splitOverHeight) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == splitOverKernel) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == clustering) {
        return DistributionMode::DUPLICATED;
    } else {
        VPUX_THROW(
                "Operation {0} was not assigned a valid multi-cluster strategy, unable to determine the distribution "
                "mode "
                "for the output tensor",
                origOp->getName());
    }
}

}  // namespace VPU
}  // namespace vpux

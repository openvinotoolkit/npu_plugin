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

inline mlir::ArrayAttr getKernelSize(mlir::Operation* origOp) {
    if (auto depthwiseConvolutionOp = mlir::dyn_cast<NCEDepthConvolutionOp>(origOp)) {
        const Shape filterShape =
                depthwiseConvolutionOp.rawFilterShape().hasValue()
                        ? Shape(parseIntArrayAttr<int64_t>(depthwiseConvolutionOp.rawFilterShape().getValue()))
                        : getShape(depthwiseConvolutionOp.filter()).toValues();
        return getIntArrayAttr(origOp->getContext(),
                               makeArrayRef({filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]}));
    } else if (auto convolutionOp = mlir::dyn_cast<NCEConvolutionOp>(origOp)) {
        const Shape filterShape = convolutionOp.rawFilterShape().hasValue()
                                          ? Shape(parseIntArrayAttr<int64_t>(convolutionOp.rawFilterShape().getValue()))
                                          : getShape(convolutionOp.filter()).toValues();
        return getIntArrayAttr(origOp->getContext(),
                               makeArrayRef({filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]}));
    } else if (auto maxPoolOp = mlir::dyn_cast<NCEMaxPoolOp>(origOp)) {
        return maxPoolOp.kernel_size();
    } else if (auto eltwiseOp = mlir::dyn_cast<NCEEltwiseOp>(origOp)) {
        return nullptr;
    } else {
        VPUX_THROW("Attempting to get kernel size for operation {0}, which is not a NCE Task", origOp->getName());
    }
}

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
NCEClusterTilingOp createDistributedTensor(ConcreteOp origOp, mlir::Value input, DistributionMode distributionMode,
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

template <class ConcreteOp>
DistributedTensorType createDistributedTensorType(ConcreteOp origOp, mlir::Value input,
                                                  DistributionMode distributionMode, mlir::ArrayAttr numTiles) {
    const auto activationTensorDistributionModeAttr = DistributionModeAttr::get(origOp.getContext(), distributionMode);

    auto kernel = getKernelSize(origOp);
    auto stride = getStride(origOp);
    auto pad = getPad(origOp);
    auto module = origOp->template getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);

    const auto numClusters = getIntAttr(origOp.getContext(), nceOp.count());

    auto distributedactivationTensorAttr = DistributedTensorAttr::get(
            activationTensorDistributionModeAttr, numTiles, kernel, pad, stride, numClusters, origOp.getContext());

    const auto shape = getShape(input);
    const auto memSpace = vpux::IndexedSymbolAttr::get(MemoryKindAttr::get(origOp.getContext(), MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(input).toAffineMap(origOp.getContext()));
    auto elemType = input.getType().template cast<vpux::NDTypeInterface>().getElementType();

    return DistributedTensorType::get(origOp.getContext(), shape.raw(), elemType, order, memSpace,
                                      distributedactivationTensorAttr);
}

template <class ConcreteOp>
mlir::ArrayAttr getActivationTensorNumTiles(ConcreteOp origOp, StringRef strategy) {
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
mlir::ArrayAttr getWeightsTensorNumTiles(ConcreteOp origOp, StringRef strategy) {
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
DistributionMode getActivationTensorDistributionMode(ConcreteOp origOp, StringRef strategy) {
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
DistributionMode getWeightsTensorDistributionMode(ConcreteOp origOp, StringRef strategy) {
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
}

template <class ConcreteOp>
DistributionMode getOutputTensorDistributionMode(ConcreteOp origOp, StringRef strategy) {
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

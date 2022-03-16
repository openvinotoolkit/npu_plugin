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

int64_t getNumberOfClustersToAvoidAlignment(int64_t outputChannels, int64_t numClustersForCompilation);
SmallVector<int64_t> getActivationTensorNumTiles(mlir::Operation* op, int64_t numClustersAvailableForCompilation,
                                                 StringRef strategy);
Optional<SmallVector<int64_t>> getActivationTensorAlignment(StringRef strategy);
SmallVector<int64_t> getOutputTensorNumTiles(mlir::Operation* op, int64_t numClustersAvailableForCompilation,
                                             StringRef strategy);
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

    if (distributionMode == DistributionMode::OVERLAPPED) {
        auto kernel = getKernelSize(origOp);
        auto stride = getStride(origOp);
        auto pad = getPad(origOp);

        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, kernel, pad, stride,
                                           numClustersAvailableForCompilation, alignment, origOp.getContext());
    } else if (distributionMode == DistributionMode::DUPLICATED) {
        int64_t numClustersToUseForLayer = numClustersAvailableForCompilation.getValue().getSExtValue();
        if (strategy == splitOverKernel) {
            auto OC = getShape(origOp->getResult(0))[Dims4D::Act::C];
            numClustersToUseForLayer = getNumberOfClustersToAvoidAlignment(
                    OC, numClustersAvailableForCompilation.getValue().getSExtValue());
        }
        optimalNumberOfClusters = mlir::IntegerAttr::get(getInt64Type(origOp->getContext()), numClustersToUseForLayer);
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, nullptr, nullptr, nullptr, nullptr,
                                           optimalNumberOfClusters, alignment, origOp.getContext());
    } else {
        const auto tileInfo = parseIntArrayAttr<int64_t>(numTiles);
        optimalNumberOfClusters = mlir::IntegerAttr::get(getInt64Type(origOp->getContext()),
                                                         *std::max_element(tileInfo.begin(), tileInfo.end()));
        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, nullptr, nullptr, nullptr,
                                           optimalNumberOfClusters, alignment, origOp.getContext());
    }

    const auto shape = getShape(input);
    const auto memSpace = vpux::IndexedSymbolAttr::get(MemoryKindAttr::get(origOp.getContext(), MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(input).toAffineMap(origOp.getContext()));
    auto elemType = input.getType().template cast<vpux::NDTypeInterface>().getElementType();

    return DistributedTensorType::get(origOp.getContext(), shape.raw(), elemType, order, memSpace,
                                      distributedActivationTensorAttr);
}

}  // namespace VPU
}  // namespace vpux

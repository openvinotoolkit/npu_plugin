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

double getChannelAlignment(double input, int64_t align);
int64_t getOptimalNumberOfClustersForSOKLayer(int64_t outputChannels, int64_t numClustersForCompilation);
SmallVector<int64_t> getActivationTensorNumTiles(mlir::Operation* op, int64_t numClustersAvailableForCompilation,
                                                 StringRef strategy);
SmallVector<int64_t> getActivationTensorAlignment(mlir::Operation* op, StringRef strategy, bool needAlignment,
                                                  mlir::ArrayAttr numTiles);
SmallVector<int64_t> getOutputTensorNumTiles(mlir::Operation* op, int64_t numClustersAvailableForCompilation,
                                             StringRef strategy);
SmallVector<int64_t> getWeightsTensorNumTiles(mlir::Operation* op, int64_t numClustersAvailableForCompilation,
                                              StringRef strategy);
SmallVector<int64_t> getWeightsTensorAlignment(mlir::Operation* op, StringRef strategy);
SmallVector<int64_t> getWeightsTableTensorNumTiles(mlir::Operation* op, int64_t numClustersAvailableForCompilation,
                                                   StringRef strategy);
SmallVector<int64_t> getActivationWindowTensorNumTiles(mlir::Operation* op, int64_t numClustersAvailableForCompilation,
                                                       StringRef strategy, ArchKind arch);
DistributionMode getActivationTensorDistributionMode(StringRef strategy);
DistributionMode getWeightsTensorDistributionMode(StringRef strategy);
DistributionMode getOutputTensorDistributionMode(StringRef strategy);
DistributionMode getActivationWindowTensorDistributionMode(StringRef strategy, ArchKind arch);
NCEClusterTilingOp createDistributedCopyOut(mlir::Operation* origOp, NCEClusterTilingOp clusterTilingOp);
mlir::ArrayAttr getKernelSize(mlir::Operation* origOp);
ShapeRef getInputShape(mlir::Operation* origOp);
int64_t getSOHPerClusterHeightAlignment(int64_t inputWidth);
bool isSplitOverHeightSupportedByDPU(ShapeRef inputShape, int64_t KY, int64_t numClusters, bool DWTypeOp);

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
                                           mlir::ArrayAttr numTiles, mlir::ArrayAttr alignment) {
    auto inputTensorDistributedTensorType =
            createDistributedTensorType(origOp, input, distributionMode, numTiles, alignment);

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
                                                  mlir::ArrayAttr alignment) {
    DistributedTensorAttr distributedActivationTensorAttr;
    auto module = origOp->template getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    const auto numClustersAvailableForCompilation = getIntAttr(origOp.getContext(), nceOp.count());
    const auto activationTensorDistributionModeAttr = DistributionModeAttr::get(origOp.getContext(), distributionMode);
    mlir::IntegerAttr optimalNumberOfClusters = numClustersAvailableForCompilation;

    auto kernel = getKernelSize(origOp);
    const auto shape = getShape(input);
    if (distributionMode == DistributionMode::OVERLAPPED) {
        auto stride = getStride(origOp);
        auto pad = getPad(origOp);

        distributedActivationTensorAttr =
                DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, kernel, pad, stride,
                                           numClustersAvailableForCompilation, alignment, origOp.getContext());
    } else if (distributionMode == DistributionMode::DUPLICATED) {
        auto OC = getShape(origOp->getResult(0))[Dims4D::Act::C];
        int64_t optimalNumClustersForLayer =
                getOptimalNumberOfClustersForSOKLayer(OC, numClustersAvailableForCompilation.getValue().getSExtValue());
        optimalNumberOfClusters =
                mlir::IntegerAttr::get(getInt64Type(origOp->getContext()), optimalNumClustersForLayer);
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

    const auto memSpace = vpux::IndexedSymbolAttr::get(MemoryKindAttr::get(origOp.getContext(), MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(input).toAffineMap(origOp.getContext()));
    auto elemType = input.getType().template cast<vpux::NDTypeInterface>().getElementType();

    return DistributedTensorType::get(origOp.getContext(), shape.raw(), elemType, order, memSpace,
                                      distributedActivationTensorAttr);
}

}  // namespace VPU
}  // namespace vpux

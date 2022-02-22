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
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/logging.hpp"
namespace vpux {

constexpr llvm::StringLiteral multiClusterStrategy = "multiClusterStrategy";
constexpr llvm::StringLiteral splitOverHeightOverLapped =
        "SplitOverHeightOverLapped";  // Strategy is for channel major convolution
constexpr llvm::StringLiteral splitOverHeight = "SplitOverHeight";
constexpr llvm::StringLiteral splitOverKernel = "SplitOverKernel";
constexpr llvm::StringLiteral clustering = "Clustering";
enum class multiClusterStrategyRange { Cluster, SplitOverH, SplitOverK, SplitOverHOverlapped };

//
// StrategyManager
//

class StrategyManager final {
public:
    explicit StrategyManager(mlir::FuncOp func, Logger log, mlir::MLIRContext* ctx);

public:
    void computeOptimalMultiClusterStrategy();
    template <class ConcreteOp>
    VPU::NCEClusterTilingOp createDistributedInputTensor(ConcreteOp& origOp, mlir::Value input,
                                                         vpux::VPU::DistributionMode distributionMode,
                                                         mlir::ArrayAttr numTiles) const;
    template <class ConcreteOp>
    vpux::VPU::DistributedTensorType createDistributedOutputTensorType(ConcreteOp& origOp,
                                                                       vpux::VPU::DistributionMode distributionMode,
                                                                       mlir::ArrayAttr numTiles) const;
    template <class ConcreteOp>
    VPU::DistributionMode getActivationTensorDistributionMode(ConcreteOp origOp) const;
    template <class ConcreteOp>
    VPU::DistributionMode getWeightsTensorDistributionMode(ConcreteOp origOp) const;
    template <class ConcreteOp>
    mlir::ArrayAttr getActivationTensorNumTiles(ConcreteOp origOp) const;
    template <class ConcreteOp>
    mlir::ArrayAttr getWeightsTensorNumTiles(ConcreteOp origOp) const;
    void removeStrategyAttribute();

private:
    template <class ConcreteOp>
    bool isOperationSplitOverHeightCompatible(ConcreteOp op) const;
    template <class ConcreteOp>
    bool isOperationSplitOverKernelCompatible(ConcreteOp op) const;
    template <class ConcreteOp>
    void assignMultiClusterStrategyForEltwiseAndMaxPool(ConcreteOp& op) const;
    template <class ConcreteOp>
    bool isOperationMultiClusterCompatible(ConcreteOp op) const;
    template <class ConcreteOp>
    double getOperationSOHEfficiency(ConcreteOp op) const;
    template <class ConcreteOp>
    double getOperationSOKEfficiency(ConcreteOp op) const;
    template <class ConcreteOp>
    double depthwiseConvolutionTotalDataTransfer(ConcreteOp origOp, const llvm::StringRef strategy) const;
    double dpuComputeTime(mlir::Operation* op, multiClusterStrategyRange Strategy);
    double dmaTime(mlir::Operation* op, multiClusterStrategyRange Strategy);

    double computeLayerSplitOverHeightEfficency(mlir::Operation* op) const;
    double computeLayerSplitOverKernelEfficency(mlir::Operation* op) const;
    double computeChannelMajorConvolutionSplitOverHeightEfficency(VPU::NCEConvolutionOp origOp) const;
    double computeZMajorConvolutionSplitOverHeightEfficency(mlir::Operation* op) const;
    double computeZMajorConvolutionSplitOverKernelEfficency(mlir::Operation* op) const;
    void setOperationStrategy(const llvm::StringRef strategy, mlir::Operation* origOp) const;
    void assignMultiClusterStrategy(mlir::Operation* op) const;
    bool isSOHandSOKEfficiencyEqual(mlir::Operation* origOp) const;

    mlir::ArrayAttr getKernelSize(mlir::Operation* origOp) const;
    mlir::ArrayAttr getStride(mlir::Operation* origOp) const;
    vpux::VPU::PaddingAttr getPad(mlir::Operation* origOp) const;
    std::map<int64_t, std::map<int64_t, double>> channelMajorEfficiencyTable() const;
    std::map<int64_t, std::map<int64_t, double>> depthwiseEfficiencyTable() const;

    const long int _minimumHeightForSOH = 20;
    const long int _minimumOutputChannelsPerCluster = 16;
    std::map<mlir::Operation*, double> _splitOverHeightEfficencies;
    std::map<mlir::Operation*, double> _splitOverKernelEfficencies;
    Logger _log;
    int32_t _numClusters;
    size_t _numDPUPerCluster;
    size_t _numDPU;
    size_t _numChannelAlignment = 16;  // TODO: read this from some hardware config
    mlir::FuncOp _func;
    mlir::MLIRContext* _ctx;
    double CMX_BANDWIDTH_;
    double DDR_BANDWIDTH_;
    double LATENCY_CMX_;
    double LATENCY_DDR_;
};

template <class ConcreteOp>
bool StrategyManager::isOperationSplitOverHeightCompatible(ConcreteOp op) const {
    const auto outputShape = getShape(op.output());
    const auto OH = outputShape[Dims4D::Act::H];
    return OH >= _minimumHeightForSOH;
}

template <class ConcreteOp>
bool StrategyManager::isOperationSplitOverKernelCompatible(ConcreteOp op) const {
    const auto outputShape = getShape(op.output());
    const auto OC = outputShape[Dims4D::Act::C];
    return OC >= _minimumOutputChannelsPerCluster * _numClusters;
}

template <class ConcreteOp>
bool StrategyManager::isOperationMultiClusterCompatible(ConcreteOp op) const {
    if (!isOperationSplitOverHeightCompatible<ConcreteOp>(op) &&
        !isOperationSplitOverKernelCompatible<ConcreteOp>(op)) {
        return false;
    } else {
        return true;
    }
}

template <class ConcreteOp>
double StrategyManager::getOperationSOHEfficiency(ConcreteOp op) const {
    if (_splitOverHeightEfficencies.find(op) != _splitOverHeightEfficencies.end()) {
        return _splitOverHeightEfficencies.find(op)->second;
    } else {
        return 0;
    }
}

template <class ConcreteOp>
double StrategyManager::getOperationSOKEfficiency(ConcreteOp op) const {
    if (_splitOverKernelEfficencies.find(op) != _splitOverKernelEfficencies.end()) {
        return _splitOverKernelEfficencies.find(op)->second;
    } else {
        return 0;
    }
}

// Initially for the greedy cost model, which assigns a strategy to a layer in isolation, without considering
// neighbouring layers. Eltwise and Maxpool should be assigned SOH, if the layer (1) fits in CMX without tiling and (2)
// it is SOH compatible i.e. OH > 20
// Otherwise it should have clustering strategy
template <class ConcreteOp>
void StrategyManager::assignMultiClusterStrategyForEltwiseAndMaxPool(ConcreteOp& op) const {
    if (isOperationSplitOverHeightCompatible<ConcreteOp>(op)) {
        op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "SplitOverHeight"));
        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'", op->getAttr(multiClusterStrategy),
                   op->getName());
    } else {
        op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "Clustering"));
        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'", op->getAttr(multiClusterStrategy),
                   op->getName());
    }
};

template <class ConcreteOp>
VPU::NCEClusterTilingOp StrategyManager::createDistributedInputTensor(ConcreteOp& origOp, mlir::Value input,
                                                                      vpux::VPU::DistributionMode distributionMode,
                                                                      mlir::ArrayAttr numTiles) const {
    const auto activationTensorDistributionModeAttr =
            vpux::VPU::DistributionModeAttr::get(origOp.getContext(), distributionMode);

    auto kernel = getKernelSize(origOp.getOperation());
    auto stride = getStride(origOp);
    auto pad = getPad(origOp);
    const auto numClusters = getIntAttr(origOp.getContext(), _numClusters);

    auto activationTensorDistributedTensorAttr = vpux::VPU::DistributedTensorAttr::get(
            activationTensorDistributionModeAttr, numTiles, kernel, pad, stride, numClusters, origOp.getContext());

    const auto inputShape = getShape(input);
    const auto memSpace =
            vpux::IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(origOp.getContext(), VPU::MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(input).toAffineMap(origOp.getContext()));
    auto elemType = input.getType().template cast<mlir::ShapedType>().getElementType();

    const auto activationTensorDistributedTensorType = vpux::VPU::DistributedTensorType::get(
            origOp.getContext(), inputShape.raw(), elemType, order, memSpace, activationTensorDistributedTensorAttr);

    _log.trace("Wrap copy operation for activation tensor for into NCEClusterTilingOp for operation");

    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(origOp, &builderLog);
    builder.setInsertionPoint(origOp);
    const auto activationTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                 mlir::ValueRange newOperands) {
        const auto memSpace = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
        auto activationTensorDistributedCopyOp = builder.create<IE::CopyOp>(origOp->getLoc(), newOperands[0], memSpace);
        builder.create<VPU::YieldOp>(loc, activationTensorDistributedCopyOp->getResults());
    };

    auto distributedActivationCopyOp = builder.create<VPU::NCEClusterTilingOp>(
            origOp->getLoc(), activationTensorDistributedTensorType, input, activationTensorBodyBuilder);

    return distributedActivationCopyOp;
}

template <class ConcreteOp>
vpux::VPU::DistributedTensorType StrategyManager::createDistributedOutputTensorType(
        ConcreteOp& origOp, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles) const {
    const auto outputTensorDistributionModeAttr =
            vpux::VPU::DistributionModeAttr::get(origOp.getContext(), distributionMode);

    auto kernel = getKernelSize(origOp.getOperation());
    auto stride = getStride(origOp);
    auto pad = getPad(origOp);
    const auto numClusters = getIntAttr(origOp.getContext(), _numClusters);

    auto outputTensorDistributedTensorAttr = vpux::VPU::DistributedTensorAttr::get(
            outputTensorDistributionModeAttr, numTiles, kernel, pad, stride, numClusters, origOp.getContext());

    const auto outputShape = getShape(origOp.output());
    const auto memSpace =
            vpux::IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(origOp.getContext(), VPU::MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(origOp.output()).toAffineMap(origOp.getContext()));
    auto elemType = origOp.output().getType().template cast<mlir::ShapedType>().getElementType();

    return vpux::VPU::DistributedTensorType::get(origOp.getContext(), outputShape.raw(), elemType, order, memSpace,
                                                 outputTensorDistributedTensorAttr);
}

template <class ConcreteOp>
mlir::ArrayAttr StrategyManager::getActivationTensorNumTiles(ConcreteOp origOp) const {
    const llvm::StringRef strategy =
            origOp->template getAttr(multiClusterStrategy).template cast<mlir::StringAttr>().getValue();
    if (strategy == splitOverHeightOverLapped) {
        return getIntArrayAttr(_ctx, makeArrayRef({1, 1, _numClusters, 1}));
    } else if (strategy == splitOverHeight) {
        return getIntArrayAttr(_ctx, makeArrayRef({1, 1, _numClusters, 1}));
    } else if (strategy == splitOverKernel) {
        return getIntArrayAttr(_ctx, makeArrayRef({1, 1, 1, 1}));
    } else if (strategy == clustering) {
        return getIntArrayAttr(_ctx, makeArrayRef({1, 1, 1, 1}));
    } else {
        VPUX_THROW(
                "Operation {0} was not assigned a valid multi-cluster strategy, unable to determine a number of tiles "
                "for the activation tensor",
                origOp->getName());
    }
}

template <class ConcreteOp>
double StrategyManager::depthwiseConvolutionTotalDataTransfer(ConcreteOp origOp, const llvm::StringRef strategy) const {
    const auto inputShape = getShape(origOp.input());
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
    const auto IC = inputShape[Dims4D::Act::C];
    const auto IH = inputShape[Dims4D::Act::H];
    const auto IW = inputShape[Dims4D::Act::W];
    const double KY = filterShape[Dims4D::Filter::KY];
    const double KX = filterShape[Dims4D::Filter::KX];
    const double WOC = filterShape[Dims4D::Filter::OC];
    const double inputTensorVolume = IC * IH * IW;
    const double weightTensorVolume = WOC * 1 * std::ceil((1 * KY * KX) / 16) * 16;  // TODO: Add precision

    if (strategy == splitOverHeight) {
        return inputTensorVolume + (_numClusters * weightTensorVolume);
    } else if (strategy == splitOverKernel) {
        return ((_numClusters * inputTensorVolume) + weightTensorVolume);
    } else {
        VPUX_THROW("Operation {0} was not assigned a valid multi-cluster strategy, unable to determine total data "
                   "movement ",
                   origOp->getName());
    }
}

template <class ConcreteOp>
mlir::ArrayAttr StrategyManager::getWeightsTensorNumTiles(ConcreteOp origOp) const {
    const llvm::StringRef strategy =
            origOp->template getAttr(multiClusterStrategy).template cast<mlir::StringAttr>().getValue();
    if (strategy == splitOverHeightOverLapped) {
        return getIntArrayAttr(_ctx, makeArrayRef({1, 1, 1, 1}));
    } else if (strategy == splitOverHeight) {
        return getIntArrayAttr(_ctx, makeArrayRef({1, 1, 1, 1}));
    } else if (strategy == splitOverKernel) {
        return getIntArrayAttr(_ctx, makeArrayRef({1, 1, _numClusters, 1}));
    } else if (strategy == clustering) {
        return getIntArrayAttr(_ctx, makeArrayRef({1, 1, 1, 1}));
    } else {
        VPUX_THROW(
                "Operation {0} was not assigned a valid multi-cluster strategy, unable to determine a number of tiles "
                "for the weights tensor",
                origOp->getName());
    }
}

template <class ConcreteOp>
VPU::DistributionMode StrategyManager::getActivationTensorDistributionMode(ConcreteOp origOp) const {
    const llvm::StringRef strategy =
            origOp->template getAttr(multiClusterStrategy).template cast<mlir::StringAttr>().getValue();
    if (strategy == splitOverHeightOverLapped) {
        return VPU::DistributionMode::overlapped;
    } else if (strategy == splitOverHeight) {
        return VPU::DistributionMode::segmented;
    } else if (strategy == splitOverKernel) {
        return VPU::DistributionMode::multicasted;
    } else if (strategy == clustering) {
        return VPU::DistributionMode::multicasted;
    } else {
        VPUX_THROW("Operation {0} was not assigned a valid multi-cluster strategy, unable to determine a distribution "
                   "mode "
                   "for the activation tensor",
                   origOp->getName());
    }
}

template <class ConcreteOp>
VPU::DistributionMode StrategyManager::getWeightsTensorDistributionMode(ConcreteOp origOp) const {
    const llvm::StringRef strategy =
            origOp->template getAttr(multiClusterStrategy).template cast<mlir::StringAttr>().getValue();
    if (strategy == splitOverHeightOverLapped) {
        return VPU::DistributionMode::multicasted;
    } else if (strategy == splitOverHeight) {
        return VPU::DistributionMode::multicasted;
    } else if (strategy == splitOverKernel) {
        return VPU::DistributionMode::segmented;
    } else if (strategy == clustering) {
        return VPU::DistributionMode::multicasted;
    } else {
        VPUX_THROW("Operation {0} was not assigned a valid multi-cluster strategy, unable to determine a distribution "
                   "mode "
                   "for the weights tensor",
                   origOp->getName());
    }
}

}  // namespace vpux

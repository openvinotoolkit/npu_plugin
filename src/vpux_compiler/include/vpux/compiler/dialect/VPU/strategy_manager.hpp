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
constexpr llvm::StringLiteral splitOverHeightOverLappedStrategy =
        "SplitOverHeightOverLapped";  // This strategy is for channel major convolutions
constexpr llvm::StringLiteral splitOverHeightStrategy = "SplitOverHeight";
constexpr llvm::StringLiteral splitOverKernelStrategy = "SplitOverKernel";

//
// StrategyManager
//

class StrategyManager final {
public:
    explicit StrategyManager(mlir::FuncOp func, Logger log);

public:
    void computeOptimalMultiClusterStrategy();
    template <class ConcreteOp>
    VPU::NCEClusterTilingOp createDistributedActivationTensor(ConcreteOp& origOp,
                                                              vpux::VPU::DistributionMode distributionMode,
                                                              mlir::ArrayAttr numTiles) const;
    template <class ConcreteOp>
    VPU::NCEClusterTilingOp createDistributedWeightsTensor(ConcreteOp& origOp,
                                                           vpux::VPU::DistributionMode distributionMode,
                                                           mlir::ArrayAttr numTiles) const;
    template <class ConcreteOp>
    vpux::VPU::DistributedTensorType createDistributedOutputTensorType(ConcreteOp& origOp,
                                                                       vpux::VPU::DistributionMode distributionMode,
                                                                       mlir::ArrayAttr numTiles) const;
    VPU::NCEClusterTilingOp createDistributedActivationTensorforMaxPool(VPU::NCEMaxPoolOp& origOp,
                                                                        vpux::VPU::DistributionMode distributionMode,
                                                                        mlir::ArrayAttr numTiles) const;

    VPU::DistributedTensorType createDistributedOutputTensorTypeforMaxPool(VPU::NCEMaxPoolOp& origOp,
                                                                           vpux::VPU::DistributionMode distributionMode,
                                                                           mlir::ArrayAttr numTiles) const;
    static VPU::DistributionMode getActivationTensorDistributionMode(const llvm::StringRef multiClusterStrategy);
    static VPU::DistributionMode getWeightsTensorDistributionMode(const llvm::StringRef multiClusterStrategy);
    static llvm::ArrayRef<int32_t> getActivationTensorNumTiles(const llvm::StringRef multiClusterStrategy);
    static llvm::ArrayRef<int32_t> getWeightsTensorNumTiles(const llvm::StringRef multiClusterStrategy);

private:
    template <class ConcreteOp>
    bool isOperationSplitOverHeightCompatible(ConcreteOp op);
    template <class ConcreteOp>
    bool isOperationSplitOverKernelCompatible(ConcreteOp op);
    template <class ConcreteOp>
    void assignMultiClusterStrategyForEltwiseAndMaxPool(ConcreteOp& op);
    void assignMultiClusterStrategy(mlir::Operation* op);
    double calculateSplitOverHeightEfficency(mlir::Operation* op);
    double calculateSplitOverKernelEfficency(mlir::Operation* op);
    mlir::ArrayAttr getKernelSize(VPU::NCEDepthConvolutionOp& origOp) const;
    mlir::ArrayAttr getKernelSize(VPU::NCEConvolutionOp& origOp) const;
    mlir::ArrayAttr getKernelSize(VPU::NCEMaxPoolOp& origOp) const;

    std::map<int64_t, std::map<int64_t, double>> channelMajorEfficiencyTable();
    std::map<int64_t, std::map<int64_t, double>> depthwiseEfficiencyTable();

    const long int _minimumHeightForSOH = 20;
    const long int _minimumOutputChannelsPerCluster = 16;
    // llvm::DenseMap<mlir::Operation*, double> _splitOverHeightEfficencies;
    // llvm::DenseMap<mlir::Operation*, double> _splitOverKernelEfficencies;
    std::map<mlir::Operation*, double> _splitOverHeightEfficencies;
    std::map<mlir::Operation*, double> _splitOverKernelEfficencies;
    Logger _log;
    long int _numClusters;
    size_t _numDPUPerCluster = 5;
    size_t _numDPU;
    size_t _numChannelAlignment = 16;
    mlir::FuncOp _func;
};

template <class ConcreteOp>
bool StrategyManager::isOperationSplitOverHeightCompatible(ConcreteOp op) {
    const auto outputShape = getShape(op.output());
    const auto OH = outputShape[Dims4D::Act::H];
    return OH >= _minimumHeightForSOH;
}

template <class ConcreteOp>
bool StrategyManager::isOperationSplitOverKernelCompatible(ConcreteOp op) {
    const auto outputShape = getShape(op.output());
    const auto OC = outputShape[Dims4D::Act::C];
    return OC >= _minimumOutputChannelsPerCluster * 4;  // change this to numCluster
}

template <class ConcreteOp>
void StrategyManager::assignMultiClusterStrategyForEltwiseAndMaxPool(ConcreteOp& op) {
    // If operation is not SOH compatible, then it has to be Clustering
    if (isOperationSplitOverHeightCompatible<ConcreteOp>(op)) {
        op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "SplitOverH"));
        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'", op->getAttr(multiClusterStrategy),
                   op->getName());
    } else {
        op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "Clustering"));
        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'", op->getAttr(multiClusterStrategy),
                   op->getName());
    }
};

template <class ConcreteOp>
VPU::NCEClusterTilingOp StrategyManager::createDistributedActivationTensor(ConcreteOp& origOp,
                                                                           vpux::VPU::DistributionMode distributionMode,
                                                                           mlir::ArrayAttr numTiles) const {
    // Specify the distribution mode of the tensor  overlapped,duplicated,segmented, multicasted,
    const auto activationTensorDistributionModeAttr =
            vpux::VPU::DistributionModeAttr::get(origOp.getContext(), distributionMode);

    // Specify the kernel
    auto kernel = getKernelSize(origOp);

    const auto numClusters = getIntAttr(origOp.getContext(), _numClusters);

    // Create DistributedTensorAttr
    auto activationTensorDistributedTensorAttr =
            vpux::VPU::DistributedTensorAttr::get(activationTensorDistributionModeAttr, numTiles, kernel,
                                                  origOp.strides(), origOp.padAttr(), numClusters, origOp.getContext());

    // Specify the inputShape
    const auto inputShape = getShape(origOp.input());

    // Specify the memSpace
    const auto memSpace =
            vpux::IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(origOp.getContext(), VPU::MemoryKind::CMX_NN));

    // Specify the order
    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(origOp.input()).toAffineMap(origOp.getContext()));

    // Element type
    auto elemType = origOp.input().getType().template cast<mlir::ShapedType>().getElementType();

    // Create DistributedTensorType
    const auto activationTensorDistributedTensorType = vpux::VPU::DistributedTensorType::get(
            origOp.getContext(), inputShape.raw(), elemType, order, memSpace, activationTensorDistributedTensorAttr);

    _log.trace("Wrap copy operation for activation into NCEClusterTilingOp");

    // Create IE::Copy Op
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
            origOp->getLoc(), activationTensorDistributedTensorType, origOp.input(), activationTensorBodyBuilder);
    return distributedActivationCopyOp;
}

template <class ConcreteOp>
VPU::NCEClusterTilingOp StrategyManager::createDistributedWeightsTensor(ConcreteOp& origOp,
                                                                        vpux::VPU::DistributionMode distributionMode,
                                                                        mlir::ArrayAttr numTiles) const {
    // Specify the distribution mode of the tensor  overlapped,duplicated,segmented, multicasted,
    const auto weightsTensorDistributionModeAttr =
            vpux::VPU::DistributionModeAttr::get(origOp.getContext(), distributionMode);

    // Specify the kernel
    const auto filterShape = origOp.rawFilterShape().hasValue()
                                     ? Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShape().getValue()))
                                     : getShape(origOp.filter());
    const auto kernel = getIntArrayAttr(
            origOp.getContext(), makeArrayRef({filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]}));

    const auto numClusters = getIntAttr(origOp.getContext(), _numClusters);

    // Create DistributedTensorAttr
    auto weightsTensorDistributedTensorAttr =
            vpux::VPU::DistributedTensorAttr::get(weightsTensorDistributionModeAttr, numTiles, kernel, origOp.strides(),
                                                  origOp.padAttr(), numClusters, origOp.getContext());

    // Specify the inputShape
    const auto inputShape = getShape(origOp.filter());

    // Specify the memSpace
    const auto memSpace =
            vpux::IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(origOp.getContext(), VPU::MemoryKind::CMX_NN));

    // Specify the order
    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(origOp.input()).toAffineMap(origOp.getContext()));

    // Element type
    auto elemType = origOp.filter().getType().template cast<mlir::ShapedType>().getElementType();

    // Create DistributedTensorType
    const auto weightsTensorDistributedTensorType = vpux::VPU::DistributedTensorType::get(
            origOp.getContext(), inputShape.raw(), elemType, order, memSpace, weightsTensorDistributedTensorAttr);

    _log.trace("Wrap copy operation for weights into NCEClusterTilingOp");

    // Create IE::Copy Op
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(origOp, &builderLog);

    builder.setInsertionPoint(origOp);
    const auto weightsTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                              mlir::ValueRange newOperands) {
        const auto memSpace = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
        auto weightsTensorDistributedCopyOp = builder.create<IE::CopyOp>(origOp->getLoc(), newOperands[0], memSpace);
        builder.create<VPU::YieldOp>(loc, weightsTensorDistributedCopyOp->getResults());
    };
    auto distributedWeightsCopyOp = builder.create<VPU::NCEClusterTilingOp>(
            origOp->getLoc(), weightsTensorDistributedTensorType, origOp.filter(), weightsTensorBodyBuilder);
    return distributedWeightsCopyOp;
}

template <class ConcreteOp>
vpux::VPU::DistributedTensorType StrategyManager::createDistributedOutputTensorType(
        ConcreteOp& origOp, vpux::VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles) const {
    // Specify the distribution mode of the tensor  overlapped,duplicated,segmented, multicasted,
    const auto outputTensorDistributionModeAttr =
            vpux::VPU::DistributionModeAttr::get(origOp.getContext(), distributionMode);

    // Specify the kernel
    auto kernel = getKernelSize(origOp);

    const auto numClusters = getIntAttr(origOp.getContext(), _numClusters);

    // Create DistributedTensorAttr
    auto outputTensorDistributedTensorAttr =
            vpux::VPU::DistributedTensorAttr::get(outputTensorDistributionModeAttr, numTiles, kernel, origOp.strides(),
                                                  origOp.padAttr(), numClusters, origOp.getContext());

    // Specify the outputShape
    const auto outputShape = getShape(origOp.output());

    // Specify the memSpace
    const auto memSpace =
            vpux::IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(origOp.getContext(), VPU::MemoryKind::CMX_NN));

    // Specify the order
    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(origOp.output()).toAffineMap(origOp.getContext()));

    // Element type
    auto elemType = origOp.output().getType().template cast<mlir::ShapedType>().getElementType();

    // Create DistributedTensorType
    return vpux::VPU::DistributedTensorType::get(origOp.getContext(), outputShape.raw(), elemType, order, memSpace,
                                                 outputTensorDistributedTensorAttr);
}

}  // namespace vpux

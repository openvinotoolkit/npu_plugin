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

constexpr llvm::StringLiteral multiClusterStrategy = "multiClusterStrategy";
constexpr llvm::StringLiteral splitOverHeightOverLapped =
        "SplitOverHeightOverLapped";  // Strategy is for channel major convolution
constexpr llvm::StringLiteral splitOverHeight = "SplitOverHeight";
constexpr llvm::StringLiteral splitOverKernel = "SplitOverKernel";
constexpr llvm::StringLiteral clustering = "Clustering";

constexpr int64_t MAXPOOL_AND_ELTWISE_SOH_EFFICIENCY = 1;
constexpr int64_t MAXPOOL_AND_ELTWISE_SOK_EFFICIENCY = 0;
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
    vpux::VPU::DistributedTensorType createDistributedInputTensorType(ConcreteOp& origOp, mlir::Value input,
                                                                      vpux::VPU::DistributionMode distributionMode,
                                                                      mlir::ArrayAttr numTiles) const;
    template <class ConcreteOp>
    vpux::VPU::DistributedTensorType createDistributedOutputTensorType(ConcreteOp& origOp,
                                                                       vpux::VPU::DistributionMode distributionMode,
                                                                       mlir::ArrayAttr numTiles) const;
    template <class ConcreteOp>
    VPU::DistributionMode getActivationTensorDistributionMode(ConcreteOp& origOp) const;
    template <class ConcreteOp>
    VPU::DistributionMode getWeightsTensorDistributionMode(ConcreteOp& origOp) const;
    template <class ConcreteOp>
    mlir::ArrayAttr getActivationTensorNumTiles(ConcreteOp& origOp) const;
    template <class ConcreteOp>
    mlir::ArrayAttr getWeightsTensorNumTiles(ConcreteOp& origOp) const;
    void removeStrategyAttribute();

private:
    template <class ConcreteOp>
    bool isOperationSplitOverHeightCompatible(ConcreteOp& op) const;
    template <class ConcreteOp>
    bool isOperationSplitOverKernelCompatible(ConcreteOp& op) const;
    template <class ConcreteOp>
    void assignMultiClusterStrategyForEltwiseAndMaxPool(ConcreteOp& op) const;
    template <class ConcreteOp>
    bool isOperationMultiClusterCompatible(ConcreteOp& op) const;
    template <class ConcreteOp>
    double getOperationSOHEfficiency(ConcreteOp& op) const;
    template <class ConcreteOp>
    double getOperationSOKEfficiency(ConcreteOp& op) const;
    template <class ConcreteOp>
    double depthwiseConvolutionTotalDataTransfer(ConcreteOp& origOp, const llvm::StringRef strategy) const;
    template <class ConcreteOp>
    bool doesSplitOverHeightFitIntoCMX(ConcreteOp& origOp) const;

    double getDepthwiseEfficiencyConstant(const int64_t& kernel, const int64_t& stride) const;
    double getChannelMajorEfficiencyConstant(const int64_t& kernel, const int64_t& stride) const;
    double computeLayerSplitOverHeightEfficency(mlir::Operation* op) const;
    double computeLayerSplitOverKernelEfficency(mlir::Operation* op) const;
    double computeChannelMajorConvolutionSplitOverHeightEfficency(VPU::NCEConvolutionOp& origOp) const;
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

    std::map<mlir::Operation*, double> _splitOverHeightEfficencies;
    std::map<mlir::Operation*, double> _splitOverKernelEfficencies;
    int32_t _numClusters;
    size_t _numDPUPerCluster;
    size_t _numDPU;
    const long int _minimumHeightForSOH = 20;
    const long int _minimumOutputChannelsPerCluster = 16;
    const size_t _numChannelAlignment = 16;  // TODO: read this from some hardware config
    mlir::FuncOp _func;
    Logger _log;
    mlir::MLIRContext* _ctx;
};

// An operation is SOH compitable if it has an output height of at least 20
// The reason is because the output tensor in each cluster will only have a
// height of 5 (20/4, assuming 4 cluster compilation).
// There are 5 DPUs in a cluster so each DPU will compute at least one output line
template <class ConcreteOp>
bool StrategyManager::isOperationSplitOverHeightCompatible(ConcreteOp& op) const {
    const auto outputShape = getShape(op.output());
    const auto OH = outputShape[Dims4D::Act::H];
    return (OH >= _minimumHeightForSOH) && doesSplitOverHeightFitIntoCMX(op);
}

// An operation is SOK compitable if it has at least 64 output channels
// The reason is because on VPU 2.0 output channels must be aligned to 16
// When an operation is SOK, 1/4 of its weights are present in each cluster (assuming 4 cluster compilation)
// Therefore, 64 / 4 = 16 implies that the operation must have at least 64 output channels in order to have
// at least 16 output channel in each cluster if it is SOK
template <class ConcreteOp>
bool StrategyManager::isOperationSplitOverKernelCompatible(ConcreteOp& op) const {
    const auto outputShape = getShape(op.output());
    const auto OC = outputShape[Dims4D::Act::C];
    return OC >= _minimumOutputChannelsPerCluster * _numClusters;
}

template <class ConcreteOp>
bool StrategyManager::isOperationMultiClusterCompatible(ConcreteOp& op) const {
    return isOperationSplitOverHeightCompatible<ConcreteOp>(op) || isOperationSplitOverKernelCompatible<ConcreteOp>(op);
}

template <class ConcreteOp>
double StrategyManager::getOperationSOHEfficiency(ConcreteOp& op) const {
    if (_splitOverHeightEfficencies.find(op) != _splitOverHeightEfficencies.end()) {
        return _splitOverHeightEfficencies.find(op)->second;
    } else {
        return 0;
    }
}

template <class ConcreteOp>
double StrategyManager::getOperationSOKEfficiency(ConcreteOp& op) const {
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
    auto inputTensorDistributedTensorType = createDistributedInputTensorType(origOp, input, distributionMode, numTiles);

    _log.trace("Wrap copy operation for input tensor for into NCEClusterTilingOp for operation");

    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(origOp, &builderLog);
    builder.setInsertionPoint(origOp);
    const auto inputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::ValueRange newOperands) {
        const auto memSpace = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
        auto inputTensorDistributedCopyOp = builder.create<IE::CopyOp>(origOp->getLoc(), newOperands[0], memSpace);
        builder.create<VPU::YieldOp>(loc, inputTensorDistributedCopyOp->getResults());
    };

    auto distributedInputCopyOp = builder.create<VPU::NCEClusterTilingOp>(
            origOp->getLoc(), inputTensorDistributedTensorType, input, inputTensorBodyBuilder);

    return distributedInputCopyOp;
}

template <class ConcreteOp>
vpux::VPU::DistributedTensorType StrategyManager::createDistributedInputTensorType(
        ConcreteOp& origOp, mlir::Value input, vpux::VPU::DistributionMode distributionMode,
        mlir::ArrayAttr numTiles) const {
    const auto inputTensorDistributionModeAttr =
            vpux::VPU::DistributionModeAttr::get(origOp.getContext(), distributionMode);

    auto kernel = getKernelSize(origOp.getOperation());
    auto stride = getStride(origOp);
    auto pad = getPad(origOp);
    const auto numClusters = getIntAttr(origOp.getContext(), _numClusters);

    auto inputTensorDistributedTensorAttr = vpux::VPU::DistributedTensorAttr::get(
            inputTensorDistributionModeAttr, numTiles, kernel, pad, stride, numClusters, origOp.getContext());

    const auto inputShape = getShape(input);
    const auto memSpace =
            vpux::IndexedSymbolAttr::get(VPU::MemoryKindAttr::get(origOp.getContext(), VPU::MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(DimsOrder::fromValue(input).toAffineMap(origOp.getContext()));
    auto elemType = input.getType().template cast<mlir::ShapedType>().getElementType();

    const auto inputTensorDistributedTensorType = vpux::VPU::DistributedTensorType::get(
            origOp.getContext(), inputShape.raw(), elemType, order, memSpace, inputTensorDistributedTensorAttr);

    return inputTensorDistributedTensorType;
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
mlir::ArrayAttr StrategyManager::getActivationTensorNumTiles(ConcreteOp& origOp) const {
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
        VPUX_THROW("Operation {0} was not assigned a valid multi-cluster strategy, unable to determine the number of "
                   "tiles "
                   "for the activation tensor",
                   origOp->getName());
    }
}

template <class ConcreteOp>
double StrategyManager::depthwiseConvolutionTotalDataTransfer(ConcreteOp& origOp,
                                                              const llvm::StringRef strategy) const {
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
        VPUX_THROW("Operation {0} was not assigned a valid multi-cluster strategy, unable to determine the total data "
                   "movement required",
                   origOp->getName());
    }
}

template <class ConcreteOp>
mlir::ArrayAttr StrategyManager::getWeightsTensorNumTiles(ConcreteOp& origOp) const {
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
        VPUX_THROW("Operation {0} was not assigned a valid multi-cluster strategy, unable to determine the number of "
                   "tiles "
                   "for the weights tensor",
                   origOp->getName());
    }
}

template <class ConcreteOp>
VPU::DistributionMode StrategyManager::getActivationTensorDistributionMode(ConcreteOp& origOp) const {
    const llvm::StringRef strategy =
            origOp->template getAttr(multiClusterStrategy).template cast<mlir::StringAttr>().getValue();
    if (strategy == splitOverHeightOverLapped) {
        return VPU::DistributionMode::OVERLAPPED;
    } else if (strategy == splitOverHeight) {
        return VPU::DistributionMode::SEGMENTED;
    } else if (strategy == splitOverKernel) {
        return VPU::DistributionMode::MULTICASTED;
    } else if (strategy == clustering) {
        return VPU::DistributionMode::MULTICASTED;
    } else {
        VPUX_THROW(
                "Operation {0} was not assigned a valid multi-cluster strategy, unable to determine the distribution "
                "mode "
                "for the activation tensor",
                origOp->getName());
    }
}

template <class ConcreteOp>
VPU::DistributionMode StrategyManager::getWeightsTensorDistributionMode(ConcreteOp& origOp) const {
    const llvm::StringRef strategy =
            origOp->template getAttr(multiClusterStrategy).template cast<mlir::StringAttr>().getValue();
    if (strategy == splitOverHeightOverLapped) {
        return VPU::DistributionMode::MULTICASTED;
    } else if (strategy == splitOverHeight) {
        return VPU::DistributionMode::MULTICASTED;
    } else if (strategy == splitOverKernel) {
        return VPU::DistributionMode::SEGMENTED;
    } else if (strategy == clustering) {
        return VPU::DistributionMode::MULTICASTED;
    } else {
        VPUX_THROW(
                "Operation {0} was not assigned a valid multi-cluster strategy, unable to determine the distribution "
                "mode "
                "for the weights tensor",
                origOp->getName());
    }
}

template <class ConcreteOp>
bool StrategyManager::doesSplitOverHeightFitIntoCMX(ConcreteOp& origOp) const {
    auto activationTensorDistributionMode = VPU::DistributionMode::SEGMENTED;
    auto activationTensorNumTiles = getIntArrayAttr(_ctx, makeArrayRef({1, 1, _numClusters, 1}));
    auto weightsTensorDistributionMode = VPU::DistributionMode::MULTICASTED;
    auto weightTensorNumTiles = getIntArrayAttr(_ctx, makeArrayRef({1, 1, 1, 1}));
    auto distributedOutputTensorType =
            createDistributedOutputTensorType(origOp, activationTensorDistributionMode, activationTensorNumTiles);
    auto outputShape = getShape(origOp.output());
    auto OC = outputShape[Dims4D::Act::C];
    Byte totalMemorySize(0);
    int64_t activationWindowSize = 0;

    if (auto convolutionOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(origOp.getOperation())) {
        auto distributedActivationTensorType = createDistributedInputTensorType(
                convolutionOp, convolutionOp.input(), activationTensorDistributionMode, activationTensorNumTiles);
        auto distributeddWeightsTensorType = createDistributedInputTensorType(
                convolutionOp, convolutionOp.filter(), weightsTensorDistributionMode, weightTensorNumTiles);
        auto weightsTable = VPU::NCEInvariant::getWeightsTableSize(OC);

        if (DimsOrder::fromValue(convolutionOp.input()) == DimsOrder::NCHW) {
            // TODO: Simplify?
            const auto filterShape =
                    convolutionOp.rawFilterShape().hasValue()
                            ? Shape(parseIntArrayAttr<int64_t>(convolutionOp.rawFilterShape().getValue()))
                            : getShape(convolutionOp.filter()).toValues();
            const auto IC = filterShape[Dims4D::Filter::IC];
            const auto KY = filterShape[Dims4D::Filter::KY];
            const auto KX = filterShape[Dims4D::Filter::KX];
            const auto kernelSize = Shape{KY, KX};
            const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(convolutionOp.strides()));
            const auto SX = kernelStrides[Dims4D::Strides::X];
            auto elemType = convolutionOp.input().getType().template cast<mlir::ShapedType>().getElementType();
            activationWindowSize = VPU::NCESparsity::getActivationWindowSize(VPU::NCESparsity::Mode::CM_CONV,
                                                                             kernelSize, SX, elemType, IC);
        }

        totalMemorySize += distributedActivationTensorType.getTotalAllocSize() +
                           distributeddWeightsTensorType.getTotalAllocSize() +
                           distributedOutputTensorType.getTotalAllocSize() + weightsTable +
                           activationWindowSize * 1_Byte;
    } else if (auto depthwiseConvolutionOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(origOp.getOperation())) {
        auto distributedActivationTensorType =
                createDistributedInputTensorType(depthwiseConvolutionOp, depthwiseConvolutionOp.input(),
                                                 activationTensorDistributionMode, activationTensorNumTiles);
        auto distributeddWeightsTensorType =
                createDistributedInputTensorType(depthwiseConvolutionOp, depthwiseConvolutionOp.filter(),
                                                 weightsTensorDistributionMode, weightTensorNumTiles);
        auto weightsTable = VPU::NCEInvariant::getWeightsTableSize(OC);

        const auto filterShape =
                depthwiseConvolutionOp.rawFilterShape().hasValue()
                        ? Shape(parseIntArrayAttr<int64_t>(depthwiseConvolutionOp.rawFilterShape().getValue()))
                        : getShape(depthwiseConvolutionOp.filter()).toValues();
        const auto IC = filterShape[Dims4D::Filter::IC];
        const auto KY = filterShape[Dims4D::Filter::KY];
        const auto KX = filterShape[Dims4D::Filter::KX];
        const auto kernelSize = Shape{KY, KX};
        const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(depthwiseConvolutionOp.strides()));
        const auto SX = kernelStrides[Dims4D::Strides::X];
        auto elemType = depthwiseConvolutionOp.input().getType().template cast<mlir::ShapedType>().getElementType();
        activationWindowSize = VPU::NCESparsity::getActivationWindowSize(VPU::NCESparsity::Mode::DW_CONV, kernelSize,
                                                                         SX, elemType, IC);

        totalMemorySize += distributedActivationTensorType.getTotalAllocSize() +
                           distributeddWeightsTensorType.getTotalAllocSize() +
                           distributedOutputTensorType.getTotalAllocSize() + weightsTable +
                           activationWindowSize * 1_Byte;
    } else if (auto maxPoolOp = mlir::dyn_cast<VPU::NCEMaxPoolOp>(origOp.getOperation())) {
        auto distributedActivationTensorType = createDistributedInputTensorType(
                maxPoolOp, maxPoolOp.input(), activationTensorDistributionMode, activationTensorNumTiles);
        auto weightsTable = VPU::NCEInvariant::getWeightsTableSize(OC);

        const auto inputShape = getShape(maxPoolOp.input());
        const auto IC = inputShape[Dims4D::Act::C];
        const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(maxPoolOp.kernel_size()));
        const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(maxPoolOp.strides()));
        const auto SX = kernelStrides[Dims4D::Strides::X];
        auto elemType = maxPoolOp.input().getType().template cast<mlir::ShapedType>().getElementType();
        activationWindowSize =
                VPU::NCESparsity::getActivationWindowSize(VPU::NCESparsity::Mode::POOL, kernelSize, SX, elemType, IC);

        totalMemorySize += distributedActivationTensorType.getTotalAllocSize() +
                           distributedOutputTensorType.getTotalAllocSize() + weightsTable +
                           activationWindowSize * 1_Byte;
    } else if (auto eltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(origOp.getOperation())) {
        auto distributedInput1TensorType = createDistributedInputTensorType(
                eltwiseOp, eltwiseOp.input1(), activationTensorDistributionMode, activationTensorNumTiles);
        auto distributedInput2TensorType = createDistributedInputTensorType(
                eltwiseOp, eltwiseOp.input2(), weightsTensorDistributionMode, weightTensorNumTiles);

        totalMemorySize += distributedInput1TensorType.getTotalAllocSize() +
                           distributedInput2TensorType.getTotalAllocSize() +
                           distributedOutputTensorType.getTotalAllocSize();
    } else {
        VPUX_THROW("Attempting to get the padding for operation {0}, which is not a NCE Task", origOp->getName());
    }

    return totalMemorySize <= VPU::getTotalCMXSize(origOp.getOperation());
}

}  // namespace vpux

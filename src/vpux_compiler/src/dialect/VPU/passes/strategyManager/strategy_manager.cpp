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

#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;

LayerCostModel::LayerCostModel(mlir::FuncOp func, Logger log, int64_t numClusters): _func(func), _log(log), _numClusters(numClusters) {
    // These latency numbers inferred from KMB db v1.2
    _CMXLatency = 5;  // Cycles, attempt to capture cost accessing CMX
    // DDR latency also measured for kmb at ~100 cycles per dma
    _DDRLatency = 100;        // Cycles, attempt to capture cost of setup DMA
    _DDRBandwidth = 8 * 0.6;  // bytes per cycle times derating factor
    const auto arch = VPU::getArch(func);
    if (arch == VPU::ArchKind::KMB) {
        _CMXBandwidth = 15;  // 32 * 1.0; //bytes per cycle times derating factor
    } else if (arch == VPU::ArchKind::TBH) {
        _CMXBandwidth = 30;  // 64 * 1.0; //bytes per cycle times derating factor
    }
    // @todo include params for ma3720
}

template <class ConcreteOp>
double LayerCostModel::clusterComputeTime(ConcreteOp op, StringRef strategy) const{
    double clusterEff = computeSplitEfficiency(op, strategy);
    auto clusterOutShape = getLargestClusterOutputShape(op, strategy, _numClusters);
    size_t baseKernelCost = 0;

    if (mlir::isa<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp>(op)) {
        auto weightsShape = getShape(op->getOperand(1));
        baseKernelCost =
            weightsShape[Dims4D::Filter::KY] * weightsShape[Dims4D::Filter::KX] * weightsShape[Dims4D::Filter::IC];
    } else if (mlir::isa<VPU::NCEMaxPoolOp>(op) || mlir::isa<VPU::NCEMaxPoolOp>(operation)) {
        auto kernel = parseIntArrayAttr<int64_t>(origOp.kernel_size());
        baseKernelCost = kernel[0] * kernel[1];
    } else if (mlir::isa<VPU::NCEEltwiseOp>(op)) {
        baseKernelCost = 1;
    } else{
        VPUX_THROW("Invalid NCE operation type");
    }

    // Actually the result is MPEVolume * baseKernelCost
    return (static_cast<double>(clusterOutShape.totalSize() * baseKernelCost)) / clusterEff;
}

// Each layer calculates the cost to move it's weights (if they exist), and it's activations
// Each DMA cost is modelled as latency + size*transfer_rate
// @warning Params {op , strategy} must be matching
template <class ConcreteOp>
double LayerCostModel::dmaTime(ConcreteOp op, StringRef strategy) const{
    double weightsCycles = 0;
    double activationWindowCycles = 0;
    double inputCycles = 0;
    double outputCycles = 0;
    const auto inOrder = DimsOrder::fromValue(op.input());
    bool isCMajor = inOrder == DimsOrder::NCHW;
    
    /// Weights cost
    if (mlir::isa<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp>(op)){
        auto clusterOutShape = getLargestClusterOutputShape(op, strategy, _numClusters);
        size_t clusterOutChannels = clusterOutShape[Dims4D::Act::C];
        auto weights = op->getOperand(1);
        auto weightsShape = getShape(weights);
        const auto weightsType = weights.getType().cast<mlir::ShapedType>();
        const auto elementType = weightsType.getElementType();
        const auto elemBytes = elementType.getIntOrFloatBitWidth() / CHAR_BIT;

        // IC * KX * KY * BytesPerElement need to align to 16B for kernels
        size_t weightsSize = (clusterOutChannels *
                            alignVal<size_t>(weightsShape[Dims4D::Filter::IC] * weightsShape[Dims4D::Filter::KY] *
                                                    weightsShape[Dims4D::Filter::KX] * elemBytes,
                                            _cmxAddressAlignment));
        // WeightTable has OC entries, each entries includes sparsity/weights pointer, bias and multi/shfit quantized
        // params. The total size for those is 16 Bytes
        size_t weightTableSize = 16 * clusterOutChannels;

        // Weights and weightTable are Segmented mode under SOK,
        // only including ddr -> cmx cost
        if (strategy == splitOverKernel) {
            weightsCycles =
                    (_DDRLatency + (static_cast<double>(weightsSize + weightTableSize) * _numClusters / _DDRBandwidth));
        } else {  
            // Duplicated mode on other strategies, including ddr->cmx cost and cmx broadcast cost
            // @warning _CMXBandwidth may be not a proper metric for cmx broadcast, refer to discussion:
            // https://github.com/intel-innersource/applications.ai.vpu-accelerators.vpux-plugin/pull/682#discussion_r819458272
            weightsCycles = (_DDRLatency + (static_cast<double>(weightsSize + weightTableSize) / _DDRBandwidth)) +
                            (_CMXLatency +
                            (static_cast<double>(weightsSize + weightTableSize) * (_numClusters - 1) / _CMXBandwidth));
        }
    }

    /// ActivationWindow cost 
    /// It's always duplicated mode and only dwconv , cmconv and maxpool own it
    size_t activationWindowSize = 0;
    if (mlir::isa<VPU::NCEMaxPoolOp, VPU::NCEDepthConvolutionOp>(op) || (mlir::isa<VPU::NCEConvolutionOp>(op) && isCMajor)) {
        const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(getKernelSize(op)));
        const auto SX = Shape(parseIntArrayAttr<int64_t>(op.strides()))[Dims4D::Strides::X];
        const auto inputDType = Op.input().getType().cast<mlir::ShapedType>().getElementType();
        auto sparsityMode = VPU::NCESparsity::Mode::DW_CONV;
        auto IC = 1;
        if (mlir::isa<VPU::NCEMaxPoolOp>(op)){
            sparsityMode = VPU::NCESparsity::Mode::POOL;
        } else if(mlir::isa<VPU::NCEConvolutionOp>(op)){
            sparsityMode = VPU::NCESparsity::Mode::CM_CONV;
            IC = getShape(op.input())[Dims4D::Act::C];
        }
        activationWindowSize = VPU::NCESparsity::getActivationWindowSize(
                sparsityMode, kernelSize, SX, inputDType, IC);
    }
    activationWindowCycles =
            (_DDRLatency + (static_cast<double>(activationWindowSize) / _DDRBandwidth)) +
            (_CMXLatency + (static_cast<double>(activationWindowSize) * (_numClusters - 1) / _CMXBandwidth));

    // @todo Overhead for Spilled input should be considered in the future,
    // as we can't get spilling strategy now
    inputCycles += 0;

    // @todo Overhead for spilled output for the same reason

    // @brief This section captures the output cost to multicast to all clusters,
    // ODU multicast happens in SOK and HKSwitch strategy.
    // @warning Duplicated mode is the same with Multicast mode?
    // @warning Currently we don't have HKSwitch Mode, We should consider it once it's added in the future
    const auto outElemBytes =
            output.getType().cast<mlir::ShapedType>().getElementType().getIntOrFloatBitWidth() / CHAR_BIT;
    // Multicast_Cost = cluster_output_size[0] * (num_clusters - 1) + cluster_output_size[1] * (num_clusters - 1) +
    //                  ... + cluster_output_size[num_clusters - 1] * (num_clusters - 1)
    if (strategy == splitOverKernel) {
        auto outputSize = 0;
        auto tiledOutShapes = getPerClusterOutputShape();
        for (auto& clusterOutShape : tiledOutShapes) {
            outputSize += clusterOutShape.totalSize();
        }
        outputCycles += (_numClusters * _CMXLatency) +
                        (static_cast<double>(outputSize * outElemBytes * (_numClusters - 1)) / _CMXBandwidth);
    }

    // Total cost for single layer without spilling consideration
    return inputCycles + activationWindowCycles + weightsCycles + outputCycles;
}

BaseLayerStrategy::BaseLayerStrategy(mlir::FuncOp func, Logger log): _func(func), _log(log) {
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, ExecutorKind::NCE);
    auto dpuExec = nceOp.getSubExecutor(VPU::ExecutorKindAttr::get(module->getContext(), VPU::ExecutorKind::DPU));
    _numClusters = nceOp.count();
    _numDPUs = dpuExec.count();
    _minimumOutputHeightForSOH = _numDPUs * _numClusters;
    _layerCostModel = LayerCostModel(func, log, _numClusters);
}

// Each DPU should compute at least one output line. Therefore in order for a layer to be SOH
// compitable it must have an output height of at least the number of DPUs x the number of clusters
// specified for compilation.
// For example for 4 cluster compilation with 5 DPUs per cluster the output height must be a
// minimum of 5x4=20.
bool BaseLayerStrategy::isOperationSplitOverHeightCompatible(mlir::Operation* op) const {
    const auto outputShape = getShape(op->getResult(0));
    const auto OH = outputShape[Dims4D::Act::H];

    return OH >= _minimumOutputHeightForSOH;
}

// Each cluster should compute at least 16 output channels. Therefore in order for a layer to be SOK
// compitable it must have an output channel of at least the number of clusters x 16
// specified for compilation.
// For example for 4 cluster compilation the output channel must be a
// minimum of 4x16=64.
bool BaseLayerStrategy::isOperationSplitOverKernelCompatible(mlir::Operation* op) const {
    const auto outputShape = getShape(op->getResult(0));
    const auto OC = outputShape[Dims4D::Act::C];
    return OC >= _numChannelAlignment * _numClusters;
}

bool BaseLayerStrategy::isOperationMultiClusterCompatible(mlir::Operation* op) const {
    if (isOperationSplitOverHeightCompatible(op) && doesLayerFitIntoCMX(op, splitOverHeight)) {
        return true;
    }

    if (isOperationSplitOverKernelCompatible(op) && doesLayerFitIntoCMX(op, splitOverKernel)) {
        return true;
    }

    return false;
}

StrategyManager::StrategyManager(mlir::FuncOp func, Logger log)
        : _func(func),
          _log(log),
          _convolutionStrategy(func, log),
          _depthConvolutionStrategy(func, log),
          _maxPoolStrategy(func, log),
          _eltwiseStrategy(func, log) {
}

void StrategyManager::assignMultiClusterStrategy() {
    const auto callback = [this](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<NCEMaxPoolOp>([this](NCEMaxPoolOp origOp) {
                    if (_maxPoolStrategy.isOperationSplitOverHeightCompatible(origOp.getOperation()) &&
                        _maxPoolStrategy.doesLayerFitIntoCMX(origOp.getOperation(), splitOverHeight)) {
                        setLayerStrategy(splitOverHeight, origOp.getOperation());
                    } else if (_maxPoolStrategy.doesLayerFitIntoCMX(origOp.getOperation(), clustering)) {
                        setLayerStrategy(clustering, origOp.getOperation());
                    }
                })
                .Case<NCEEltwiseOp>([this](NCEEltwiseOp origOp) {
                    if (_eltwiseStrategy.isOperationSplitOverHeightCompatible(origOp.getOperation()) &&
                        _eltwiseStrategy.doesLayerFitIntoCMX(origOp.getOperation(), splitOverHeight)) {
                        setLayerStrategy(splitOverHeight, origOp.getOperation());
                    } else if (_eltwiseStrategy.doesLayerFitIntoCMX(origOp.getOperation(), clustering)) {
                        setLayerStrategy(clustering, origOp.getOperation());
                    }
                })
                .Case<NCEConvolutionOp>([this](NCEConvolutionOp origOp) {
                    if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NHWC) {
                        if (_convolutionStrategy.isOperationMultiClusterCompatible(origOp.getOperation())) {
                            auto bestStrategy = _convolutionStrategy.getOptimalLayerStrategy(origOp);
                            setLayerStrategy(bestStrategy, origOp.getOperation());
                        } else if (_convolutionStrategy.doesLayerFitIntoCMX(origOp.getOperation(), clustering)) {
                            setLayerStrategy(clustering, origOp.getOperation());
                        }
                    } else if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NCHW) {
                        const auto arch = VPU::getArch(origOp.getOperation());
                        const auto canUseCMajor = VPU::NCEInvariant::isChannelMajorCompatible(
                                arch, origOp.input().getType().cast<vpux::NDTypeInterface>());

                        if (canUseCMajor &&
                            _convolutionStrategy.isOperationSplitOverHeightCompatible(origOp.getOperation()) &&
                            _convolutionStrategy.doesLayerFitIntoCMX(origOp.getOperation(),
                                                                     splitOverHeightOverlapped)) {
                            setLayerStrategy(splitOverHeightOverlapped, origOp.getOperation());
                        } else if (_convolutionStrategy.doesLayerFitIntoCMX(origOp.getOperation(), clustering)) {
                            setLayerStrategy(clustering, origOp.getOperation());
                        }
                    } else {
                        VPUX_THROW("Unsupported input layout {0} to convolution ",
                                   DimsOrder::fromValue(origOp.input()));
                    }
                })
                .Case<NCEDepthConvolutionOp>([this](NCEDepthConvolutionOp origOp) {
                    if (_depthConvolutionStrategy.isOperationMultiClusterCompatible(origOp.getOperation())) {
                        auto bestStrategy = _depthConvolutionStrategy.getOptimalLayerStrategy(origOp);
                        setLayerStrategy(bestStrategy, origOp.getOperation());
                    } else if (_depthConvolutionStrategy.doesLayerFitIntoCMX(origOp.getOperation(), clustering)) {
                        setLayerStrategy(clustering, origOp.getOperation());
                    }
                })
                .Default([this](mlir::Operation* unknownOp) -> void {
                    _log.trace("Operation '{0}' at '{1}' is not supported by the NCE therefore it should not have a "
                               "multi-cluster strategy",
                               unknownOp->getName(), unknownOp->getLoc());
                });
    };

    _func.walk(callback);
}

void StrategyManager::setLayerStrategy(StringRef strategy, mlir::Operation* origOp) const {
    if (strategy == splitOverHeightOverlapped) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverHeightOverlapped"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else if (strategy == splitOverHeight) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverHeight"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else if (strategy == splitOverKernel) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverKernel"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else if (strategy == clustering) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "Clustering"));
        _log.trace("Assigning multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else {
        VPUX_THROW("Attempting to assign an invalid strategy to operation {0}", origOp->getName());
    }
}

// The function computes the actual output tensor volume (i.e. computation that is performed)
// given the stratey and the MPE mode
double LayerCostModel::calculateMPEVolume(VPU::MPEMode mpeMode, Shape shape) const {
    int64_t mpeHeight;
    int64_t mpeWidth;
    if (mpeMode == VPU::MPEMode::VECTOR) {
        mpeHeight = 16;
        mpeWidth = 1;
    } else if (mpeMode == VPU::MPEMode::MATRIX) {
        mpeHeight = 4;
        mpeWidth = 4;
    } else {
        VPUX_THROW("Unsupported MPE mode {0}", mpeMode);
    }

    return static_cast<double>(_numDPUs * divUp((mpeHeight * divUp(shape[Dims4D::Act::H], mpeHeight) * mpeWidth *
                                                 divUp(shape[Dims4D::Act::W], mpeWidth) * _numChannelAlignment *
                                                 divUp(shape[Dims4D::Act::C], _numChannelAlignment)),
                                                _numDPUs));
}

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

// This pass assigns multi-clustering strategies to layers and converts them into NCEClusterTiling operations.
// It considers layers in isolation and computes the hardware efficiency of the layers if they
// were to be split over height (SOH) or split over kernel (SOK). It then chooses the most optimal
// stratey. A prerequisite is that the layer fits in CMX when multi-clustered. If the layer does not
// fit in CMX then it is not converted to NCEClustertiling.

StrategyManager::StrategyManager(mlir::FuncOp func, Logger log, mlir::MLIRContext* ctx)
        : _func(func), _log(log), _ctx(ctx) {
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto nceOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
    auto dpuOp = nceOp.getSubExecutor(VPU::ExecutorKindAttr::get(module->getContext(), VPU::ExecutorKind::DPU));

    _numClusters = nceOp.count();
    _numDPUPerCluster = dpuOp.count();
    _numDPU = _numClusters * _numDPUPerCluster;
}

double StrategyManager::getDepthwiseEfficiencyConstant(const int64_t& kernel, const int64_t& stride) const {
    if (depthwiseEfficiencyTable().count(kernel)) {
        auto table = depthwiseEfficiencyTable()[kernel];
        if (table.count(stride)) {
            return depthwiseEfficiencyTable()[kernel][stride];
        } else {
            VPUX_THROW("The stide size {0} does not exist in the depthwise efficiency table", stride);
        }
    } else {
        VPUX_THROW("The kernel size {0} does not exist in the depthwise efficiency table", kernel);
    }
}

double StrategyManager::getChannelMajorEfficiencyConstant(const int64_t& kernel, const int64_t& stride) const {
    if (channelMajorEfficiencyTable().count(kernel)) {
        auto table = depthwiseEfficiencyTable()[kernel];
        if (table.count(stride)) {
            return channelMajorEfficiencyTable()[kernel][stride];
        } else {
            VPUX_THROW("The stide size {0} does not exist in the channel major convolution efficiency table", stride);
        }
    } else {
        VPUX_THROW("The kernel size {0} does not exist in the channel major convolution efficiency table", kernel);
    }
}

// This depthwise convolution efficiency table is from the ArchBench tool
// It returns a h/w efficiency constant for a given stride and kernel size
std::map<int64_t, std::map<int64_t, double>> StrategyManager::depthwiseEfficiencyTable() const {
    return {{
            {3, {{1, 0.165}, {2, 0.128}, {4, 0.128}, {6, 0.165}}},
            {5, {{1, 0.483}, {2, 0.241}, {4, 0.132}, {6, 0.483}}},
            {7, {{1, 0.6}, {2, 0.2965}, {4, 0.15}, {6, 0.0395}}},
            {9, {{1, 0.8008}, {2, 0.4687}, {4, 0.2266}, {6, 0.8008}}},
            {11, {{1, 0.9023}, {2, 0.4687}, {4, 0.2366}, {6, 0.9023}}},
    }};
}

// This channel-major convolution efficiency table is from the ArchBench tool
// It returns a h/w efficiency constant for a given stride and kernel size
std::map<int64_t, std::map<int64_t, double>> StrategyManager::channelMajorEfficiencyTable() const {
    return {{
            {3, {{1, 0.253}, {2, 0.183594}, {4, 0.183594}}},
            {5, {{1, 0.535156}, {2, 0.2773}, {4, 0.152344}}},
            {7, {{1, 0.6}, {2, 0.2965}, {4, 0.15}}},
            {9, {{1, 0.8008}, {2, 0.4687}, {4, 0.2266}}},
            {11, {{1, 0.9023}, {2, 0.4687}, {4, 0.2366}}},
    }};
}

double getChannelAlignment(double input, size_t unit) {
    if (!unit) {
        VPUX_THROW("Invalid alignment to {0} requested", unit);
    }

    return std::ceil(input / unit) * unit;
}

double StrategyManager::computeChannelMajorConvolutionSplitOverHeightEfficency(VPU::NCEConvolutionOp& origOp) const {
    const auto outputShape = getShape(origOp.output());
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
    const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
    const double OC = outputShape[Dims4D::Act::C];
    const double OH = outputShape[Dims4D::Act::H];
    const double OW = outputShape[Dims4D::Act::W];
    const auto KY = filterShape[Dims4D::Filter::KY];
    auto efficiencyConstant = getChannelMajorEfficiencyConstant(KY, strides[0]);
    const double outputTensorVolume = OC * OH * OW;

    double efficency = efficiencyConstant *
                       std::max(outputTensorVolume / (getChannelAlignment(OH, _numChannelAlignment) *
                                                      getChannelAlignment(OH, _numDPU) *
                                                      getChannelAlignment(OC, _numChannelAlignment)),
                                outputTensorVolume / (getChannelAlignment(OH, _numChannelAlignment * _numClusters) *
                                                      getChannelAlignment(OW, _numDPUPerCluster) *
                                                      getChannelAlignment(OC, _numChannelAlignment)));
    return efficency;
}

// This method computes the SOH efficiency for z-major type operations
// i.e. z-major convolution and depthwise convolution
double StrategyManager::computeZMajorConvolutionSplitOverHeightEfficency(mlir::Operation* op) const {
    double efficiencyConstant = 0;
    double outputTensorVolume = 0;
    double OC;
    double OH;
    double OW;
    if (auto depthwiseConvolutionOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        const auto outputShape = getShape(depthwiseConvolutionOp.output());
        OC = outputShape[Dims4D::Act::C];
        OH = outputShape[Dims4D::Act::H];
        OW = outputShape[Dims4D::Act::W];
        const auto filterShape = Shape(parseIntArrayAttr<int64_t>(depthwiseConvolutionOp.rawFilterShapeAttr()));
        const auto strides = parseIntArrayAttr<int64_t>(depthwiseConvolutionOp.strides());
        const auto KY = filterShape[Dims4D::Filter::KY];
        efficiencyConstant = getDepthwiseEfficiencyConstant(KY, strides[0]);

    } else if (auto convolutionOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        const auto outputShape = getShape(convolutionOp.output());
        OC = outputShape[Dims4D::Act::C];
        OH = outputShape[Dims4D::Act::H];
        OW = outputShape[Dims4D::Act::W];
        efficiencyConstant = 1.0;
    } else {
        VPUX_THROW("Attempting to calculate the hardware efficiency for operation {0}, which is not a z-major "
                   "compatible operation",
                   op->getName());
    }

    outputTensorVolume = OC * OH * OW;

    return efficiencyConstant *
           std::max((outputTensorVolume / _numClusters) /
                            getChannelAlignment((getChannelAlignment(std::ceil(OH / _numClusters), _numClusters) *
                                                 getChannelAlignment(OW, _numClusters) *
                                                 getChannelAlignment(OC, _numChannelAlignment)),
                                                _numDPUPerCluster),
                    (outputTensorVolume / _numClusters) /
                            getChannelAlignment(
                                    (getChannelAlignment(std::ceil(OH / _numClusters), _numChannelAlignment) *
                                     getChannelAlignment(OW, 1) * getChannelAlignment(OC, _numChannelAlignment)),
                                    _numDPUPerCluster));
}

// This method computes the SOH efficiency for z-major type operations
// i.e. z-major convolution and depthwise convolution
double StrategyManager::computeZMajorConvolutionSplitOverKernelEfficency(mlir::Operation* op) const {
    double efficiencyConstant = 0;
    double outputTensorVolume = 0;
    double OC;
    double OH;
    double OW;
    if (auto depthwiseConvolutionOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        const auto outputShape = getShape(depthwiseConvolutionOp.output());
        OC = outputShape[Dims4D::Act::C];
        OH = outputShape[Dims4D::Act::H];
        OW = outputShape[Dims4D::Act::W];
        const auto filterShape = Shape(parseIntArrayAttr<int64_t>(depthwiseConvolutionOp.rawFilterShapeAttr()));
        const auto strides = parseIntArrayAttr<int64_t>(depthwiseConvolutionOp.strides());
        const auto KY = filterShape[Dims4D::Filter::KY];
        efficiencyConstant = depthwiseEfficiencyTable()[KY][strides[0]];

    } else if (auto convolutionOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        const auto outputShape = getShape(convolutionOp.output());
        OC = outputShape[Dims4D::Act::C];
        OH = outputShape[Dims4D::Act::H];
        OW = outputShape[Dims4D::Act::W];
        efficiencyConstant = 1.0;
    } else {
        VPUX_THROW("Attempting to calculate the hardware efficiency for operation {0}, which is not a z-major "
                   "compatible operation",
                   op->getName());
    }

    outputTensorVolume = OC * OH * OW;

    return efficiencyConstant *
           std::max((outputTensorVolume / _numClusters) /
                            getChannelAlignment(
                                    (getChannelAlignment(OH, _numClusters) * getChannelAlignment(OW, _numClusters) *
                                     getChannelAlignment(std::ceil(OC / _numClusters), _numChannelAlignment)),
                                    _numDPUPerCluster),
                    (outputTensorVolume / _numClusters) /
                            getChannelAlignment(
                                    (getChannelAlignment(OH, _numChannelAlignment) * getChannelAlignment(OW, 1) *
                                     getChannelAlignment(std::ceil(OC / _numClusters), _numChannelAlignment)),
                                    _numDPUPerCluster));
}

// This method computes the SOH hardware efficiency of all NCE operations
double StrategyManager::computeLayerSplitOverHeightEfficency(mlir::Operation* op) const {
    return llvm::TypeSwitch<mlir::Operation*, double>(op)
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp origOp) {
                // A different efficency formula is required for channel-major and z-major convolutions
                // Channel-major convolution
                double efficiency = 0;
                if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NCHW) {
                    efficiency = computeChannelMajorConvolutionSplitOverHeightEfficency(origOp);
                    // Z-major convolution
                } else {
                    efficiency = computeZMajorConvolutionSplitOverHeightEfficency(origOp.getOperation());
                }
                _log.trace("The SOH efficiency for the convolution {0} is {1}", origOp->getName(), efficiency);
                return efficiency;
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp origOp) {
                double efficiency = 1;
                _log.trace("The SOH efficiency for MaxPool operation {0} is 1 as it should be assigned SOH {1}",
                           origOp->getName(), efficiency);
                return efficiency;
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp origOp) {
                double efficiency = 1;
                _log.trace("The SOH efficiency for Eltwsie operation {0} is 1 as it should be assigned SOH {1}",
                           origOp->getName(), efficiency);
                return efficiency;
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp origOp) {
                double efficiency = computeZMajorConvolutionSplitOverHeightEfficency(origOp.getOperation());
                _log.trace("The SOH efficiency for the depthwise convolution {0} is {0}", origOp->getName(),
                           efficiency);
                return efficiency;
            });
}

// This method computes the SOK hardware efficiency of all NCE operations
double StrategyManager::computeLayerSplitOverKernelEfficency(mlir::Operation* op) const {
    return llvm::TypeSwitch<mlir::Operation*, double>(op)
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp origOp) {
                double efficiency = computeZMajorConvolutionSplitOverKernelEfficency(origOp.getOperation());
                _log.trace("The SOK efficiency for the convolution is {0}", efficiency);
                return efficiency;
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp origOp) {
                double efficiency = 1;
                _log.trace("The SOK efficiency for MaxPool operation {0} is 1 as it should be assigned SOH {1}",
                           origOp->getName(), efficiency);
                return 1;
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp origOp) {
                double efficiency = 1;
                _log.trace("The SOK efficiency for Eltwsie operation {0} is 1 as it should be assigned SOH {1}",
                           origOp->getName(), efficiency);
                return 1;
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp origOp) {
                double efficiency = computeZMajorConvolutionSplitOverKernelEfficency(origOp.getOperation());
                _log.trace("The SOK efficiency for the group convolution is {0}", efficiency);
                return efficiency;
            });
}

// This method computes the multi-cluster efficiency for an operation
void StrategyManager::computeOptimalMultiClusterStrategy() {
    const auto callback = [&](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*, void>(op)

                // MaxPool is assigned SOH if it is compitable, otherwise clustering
                .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp) {
                    assignMultiClusterStrategy(op);
                })
                // Eltwise is assigned SOH if it is compitable, otherwise clustering
                .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp op) {
                    assignMultiClusterStrategy(op);
                })
                .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp op) {
                    // Check if the operation SOH compatible
                    if (isOperationSplitOverHeightCompatible<VPU::NCEConvolutionOp>(op)) {
                        _splitOverHeightEfficencies.insert({op, computeLayerSplitOverHeightEfficency(op)});
                    }
                    // Check if the operation SOK compatible
                    if (isOperationSplitOverKernelCompatible<VPU::NCEConvolutionOp>(op)) {
                        _splitOverKernelEfficencies.insert({op, computeLayerSplitOverKernelEfficency(op)});
                    }
                    // Assign the most optimal strategy
                    assignMultiClusterStrategy(op);
                })
                .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp op) {
                    // Check if the operation SOH compatible
                    if (isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                        _splitOverHeightEfficencies.insert({op, computeLayerSplitOverHeightEfficency(op)});
                    }
                    // Check if the operation SOK compatible
                    if (isOperationSplitOverKernelCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                        _splitOverKernelEfficencies.insert({op, computeLayerSplitOverKernelEfficency(op)});
                    }
                    // Assign the most optimal strategy
                    assignMultiClusterStrategy(op);
                });
    };

    _func.walk(callback);
}

void StrategyManager::assignMultiClusterStrategy(mlir::Operation* op) const {
    llvm::TypeSwitch<mlir::Operation*, void>(op)
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp op) {
                // If operation is neither SOH or SOK compatible, then assign clustering strategy
                if (!isOperationMultiClusterCompatible<VPU::NCEConvolutionOp>(op)) {
                    setOperationStrategy(clustering, op.getOperation());
                }
                // Compare the SOH and SOK efficiencies and select SOH if they are equal
                else if (getOperationSOHEfficiency(op) >= getOperationSOKEfficiency(op)) {
                    // A channel major convolution should be have SplitOverHOverLapped strategy
                    if (DimsOrder::fromValue(op.input()) == DimsOrder::NCHW) {
                        // TODO: Enable channel-major to be converted to NCEClusterTiling
                        // when splitOverHeightOverLapped is supported in unrolling
                        setOperationStrategy(splitOverHeightOverLapped, op.getOperation());
                        // A Z-major convolution
                    } else {
                        setOperationStrategy(splitOverHeight, op.getOperation());
                    }
                }
                // Else SOK is more efficient than SOH
                else if (getOperationSOHEfficiency(op) < getOperationSOKEfficiency(op)) {
                    setOperationStrategy(splitOverKernel, op.getOperation());
                }
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp op) {
                // If operation is neither SOH or SOK compatible, then it has clustering strategy
                if (!isOperationMultiClusterCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    setOperationStrategy(clustering, op.getOperation());
                }
                // If the SOH and SOK efficiency values are equal and the operation is both SOH and SOK compatible
                // then select either SOK or SOH based on the lesser amount of data that has to be moved
                else if (isSOHandSOKEfficiencyEqual(op.getOperation()) &&
                         isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op) &&
                         isOperationSplitOverKernelCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    // Is the total data transfer for SOK less than SOH
                    if (depthwiseConvolutionTotalDataTransfer<VPU::NCEDepthConvolutionOp>(op, splitOverKernel) <
                        depthwiseConvolutionTotalDataTransfer<VPU::NCEDepthConvolutionOp>(op, splitOverHeight)) {
                        setOperationStrategy(splitOverKernel, op.getOperation());
                    } else {
                        setOperationStrategy(splitOverHeight, op.getOperation());
                    }
                    // SOH is more efficent than SOK
                } else if ((getOperationSOHEfficiency(op) > getOperationSOKEfficiency(op)) &&
                           isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    setOperationStrategy(splitOverHeight, op.getOperation());

                    // SOH is more efficent than SOK but the operation is not SOH compitable, then it has clustering
                    // strategy
                } else if ((getOperationSOHEfficiency(op) > getOperationSOKEfficiency(op)) &&
                           !isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    setOperationStrategy(clustering, op.getOperation());
                    // SOK is more efficent than SOH
                } else if ((getOperationSOHEfficiency(op) < getOperationSOKEfficiency(op)) &&
                           isOperationSplitOverKernelCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    setOperationStrategy(splitOverKernel, op.getOperation());
                    // SOK is more efficent than SOH but the operation is not SOK compitable, then it has clustering
                    // strategy
                } else if ((getOperationSOHEfficiency(op) < getOperationSOKEfficiency(op)) &&
                           !isOperationSplitOverKernelCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    setOperationStrategy(clustering, op.getOperation());
                }
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp op) {
                assignMultiClusterStrategyForEltwiseAndMaxPool<VPU::NCEMaxPoolOp>(op);
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp op) {
                assignMultiClusterStrategyForEltwiseAndMaxPool<VPU::NCEEltwiseOp>(op);
            });
}

mlir::ArrayAttr StrategyManager::getKernelSize(mlir::Operation* origOp) const {
    if (auto depthwiseConvolutionOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(origOp)) {
        const Shape filterShape =
                depthwiseConvolutionOp.rawFilterShape().hasValue()
                        ? Shape(parseIntArrayAttr<int64_t>(depthwiseConvolutionOp.rawFilterShape().getValue()))
                        : getShape(depthwiseConvolutionOp.filter()).toValues();
        return getIntArrayAttr(const_cast<mlir::FuncOp&>(_func).getContext(),
                               makeArrayRef({filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]}));
    } else if (auto convolutionOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(origOp)) {
        const Shape filterShape = convolutionOp.rawFilterShape().hasValue()
                                          ? Shape(parseIntArrayAttr<int64_t>(convolutionOp.rawFilterShape().getValue()))
                                          : getShape(convolutionOp.filter()).toValues();
        return getIntArrayAttr(const_cast<mlir::FuncOp&>(_func).getContext(),
                               makeArrayRef({filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]}));
    } else if (auto maxPoolOp = mlir::dyn_cast<VPU::NCEMaxPoolOp>(origOp)) {
        return maxPoolOp.kernel_size();
    } else if (auto eltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(origOp)) {
        return nullptr;
    } else {
        VPUX_THROW("Attempting to get kernel for operation {0}, which is not a NCE Task", origOp->getName());
    }
}

mlir::ArrayAttr StrategyManager::getStride(mlir::Operation* origOp) const {
    if (auto depthwiseConvolutionOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(origOp)) {
        return depthwiseConvolutionOp.strides();
    } else if (auto convolutionOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(origOp)) {
        return convolutionOp.strides();
    } else if (auto maxPoolOp = mlir::dyn_cast<VPU::NCEMaxPoolOp>(origOp)) {
        return maxPoolOp.strides();
    } else if (auto eltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(origOp)) {
        return nullptr;
    } else {
        VPUX_THROW("Attempting to get stride for operation {0}, which is not a NCE Task", origOp->getName());
    }
}

vpux::VPU::PaddingAttr StrategyManager::getPad(mlir::Operation* origOp) const {
    if (auto depthwiseConvolutionOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(origOp)) {
        return depthwiseConvolutionOp.padAttr();
    } else if (auto convolutionOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(origOp)) {
        return convolutionOp.padAttr();
    } else if (auto maxPoolOp = mlir::dyn_cast<VPU::NCEMaxPoolOp>(origOp)) {
        return maxPoolOp.padAttr();
    } else if (auto eltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(origOp)) {
        return nullptr;
    } else {
        VPUX_THROW("Attempting to get pad for operation {0}, which is not a NCE Task", origOp->getName());
    }
}

void StrategyManager::removeStrategyAttribute() {
    _func->walk([](mlir::Operation* op) {
        if (op->hasAttr(multiClusterStrategy)) {
            op->removeAttr(multiClusterStrategy);
        }
    });
}

bool StrategyManager::isSOHandSOKEfficiencyEqual(mlir::Operation* origOp) const {
    if (_splitOverHeightEfficencies.find(origOp)->second == _splitOverKernelEfficencies.find(origOp)->second) {
        return true;
    } else {
        return false;
    }
}

void StrategyManager::setOperationStrategy(const llvm::StringRef strategy, mlir::Operation* origOp) const {
    if (strategy == splitOverHeightOverLapped) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverHeightOverLapped"));
        _log.trace("Assiging multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else if (strategy == splitOverHeight) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverHeight"));
        _log.trace("Assiging multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else if (strategy == splitOverKernel) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverKernel"));
        _log.trace("Assiging multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else if (strategy == clustering) {
        origOp->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "Clustering"));
        _log.trace("Assiging multi-cluster strategy '{0}' to layer '{1}'", strategy, origOp->getName());
    } else {
        VPUX_THROW("Attempting to assing an invalid strategy to operation {0}", origOp->getName());
    }
}
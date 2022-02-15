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
// they were to be split over height (SOH) or split over kernel (SOK). It then chooses the more optimal
// stratey. A prerequisite is that the layer fits in CMX when multi-clustered. If the layer does not
// fit in CMX then it is not converted to NCEClustertiling.

StrategyManager::StrategyManager(mlir::FuncOp func, Logger log, mlir::MLIRContext* ctx)
        : _log(log), _func(func), _ctx(ctx) {
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto nceOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
    auto dpuOp = nceOp.getSubExecutor(VPU::ExecutorKindAttr::get(module->getContext(), VPU::ExecutorKind::DPU));

    _numClusters = 4;  // TODO: = nceOp.count(); // returns 1

    _numDPUPerCluster = dpuOp.count();

    _numDPU = _numClusters * _numDPUPerCluster;
}

// This channel major efficiency table is from the ArchBench tool
// It returns a h/w efficiency constant for a given stride and kernel size
std::map<int64_t, std::map<int64_t, double>> StrategyManager::depthwiseEfficiencyTable() {
    return {{
            {3, {{1, 0.165}, {2, 0.128}, {4, 0.128}, {6, 0.165}}},
            {5, {{1, 0.483}, {2, 0.241}, {4, 0.132}, {6, 0.483}}},
            {7, {{1, 0.6}, {2, 0.2965}, {4, 0.15}, {6, 0.0395}}},
            {9, {{1, 0.8008}, {2, 0.4687}, {4, 0.2266}, {6, 0.8008}}},
            {11, {{1, 0.9023}, {2, 0.4687}, {4, 0.2366}, {6, 0.9023}}},
    }};
}

// This depthwise convolution efficiency table is from the ArchBench tool
// It returns a h/w efficiency constant for a given stride and kernel size
std::map<int64_t, std::map<int64_t, double>> StrategyManager::channelMajorEfficiencyTable() {
    return {{
            {3, {{1, 0.253}, {2, 0.183594}, {4, 0.183594}}},
            {5, {{1, 0.535156}, {2, 0.2773}, {4, 0.152344}}},
            {7, {{1, 0.6}, {2, 0.2965}, {4, 0.15}}},
            {9, {{1, 0.8008}, {2, 0.4687}, {4, 0.2266}}},
            {11, {{1, 0.9023}, {2, 0.4687}, {4, 0.2366}}},
    }};
}

double getChannelAlignment(double input, size_t unit) {
    return std::ceil(input / unit) * unit;
}

double StrategyManager::computeChannelMajorConvolutionSplitOverHeightEfficency(VPU::NCEConvolutionOp origOp) {
    const auto outputShape = getShape(origOp.output().getType().cast<mlir::ShapedType>());
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
    const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
    const double OC = outputShape[Dims4D::Act::C];
    const double OH = outputShape[Dims4D::Act::H];
    const double OW = outputShape[Dims4D::Act::W];
    const auto KY = filterShape[Dims4D::Filter::KY];
    auto efficiencyConstant = channelMajorEfficiencyTable()[KY][strides[0]];
    const double outputTensorVolume = OC * OH * OW;

    // TODO: Can this be simpler?
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
double StrategyManager::computeZMajorConvolutionSplitOverHeightEfficency(mlir::Operation* op) {
    double efficiencyConstant = 0;
    double outputTensorVolume = 0;
    double OC;
    double OH;
    double OW;
    if (auto depthwiseConvolutionOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        const auto outputShape = getShape(depthwiseConvolutionOp.output().getType().cast<mlir::ShapedType>());
        OC = outputShape[Dims4D::Act::C];
        OH = outputShape[Dims4D::Act::H];
        OW = outputShape[Dims4D::Act::W];
        const auto filterShape = Shape(parseIntArrayAttr<int64_t>(depthwiseConvolutionOp.rawFilterShapeAttr()));
        const auto strides = parseIntArrayAttr<int64_t>(depthwiseConvolutionOp.strides());
        const auto KY = filterShape[Dims4D::Filter::KY];
        efficiencyConstant = depthwiseEfficiencyTable()[KY][strides[0]];

    } else if (auto convolutionOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        const auto outputShape = getShape(convolutionOp.output().getType().cast<mlir::ShapedType>());
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
// This method computes the SOH hardware efficiency of all NCE operations
double StrategyManager::computeLayerSplitOverHeightEfficency(mlir::Operation* op) {
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
double StrategyManager::computeLayerSplitOverKernelEfficency(mlir::Operation* op) {
    const auto splitOverKernelFormula = [&](double OH, double OW, double OC) {
        double outputTensorVolume = OC * OH * OW;
        return std::max((outputTensorVolume / _numClusters) /
                                getChannelAlignment(
                                        (getChannelAlignment(OH, _numClusters) * getChannelAlignment(OW, _numClusters) *
                                         getChannelAlignment(std::ceil(OC / _numClusters), _numChannelAlignment)),
                                        _numDPUPerCluster),
                        (outputTensorVolume / _numClusters) /
                                getChannelAlignment(
                                        (getChannelAlignment(OH, _numChannelAlignment) * getChannelAlignment(OW, 1) *
                                         getChannelAlignment(std::ceil(OC / _numClusters), _numChannelAlignment)),
                                        _numDPUPerCluster));
    };

    return llvm::TypeSwitch<mlir::Operation*, double>(op)
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp origOp) {
                const auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
                const auto outputShape = getShape(outputType);
                const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
                const double OC = outputShape[Dims4D::Act::C];
                const double OH = outputShape[Dims4D::Act::H];
                const double OW = outputShape[Dims4D::Act::W];
                const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
                double efficency = splitOverKernelFormula(OH, OW, OC);
                _log.trace("The SOK efficiency for the convolution is {0}", efficency);
                return efficency;
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp) {
                return 1;
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp) {
                _log.trace("Eltwise {0} operation detected. SOH wil be assigned", op->getName());
                return 1;
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp origOp) {
                const auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
                const auto outputShape = getShape(outputType);
                const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
                const double OC = outputShape[Dims4D::Act::C];
                const double OH = outputShape[Dims4D::Act::H];
                const double OW = outputShape[Dims4D::Act::W];
                const auto KY = filterShape[Dims4D::Filter::KY];
                const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());

                auto efficiencyConstant = depthwiseEfficiencyTable()[KY][strides[0]];

                double efficency = efficiencyConstant * splitOverKernelFormula(OH, OW, OC);
                _log.trace("The SOK efficiency for the group convolution is {0}", efficency);
                return efficency;
            });
}

// This method computes the multi-cluster efficiency value for operation
// If it is not compatible with the multi-cluster strategy, then the efficiency is 0
void StrategyManager::computeOptimalMultiClusterStrategy() {
    const auto callback = [&](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp) {
                    assignMultiClusterStrategy(op);
                })
                .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp op) {
                    assignMultiClusterStrategy(op);
                })
                .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp op) {
                    // Check if the operation SOH compatible, otherwise set the efficiency to 0
                    if (isOperationSplitOverHeightCompatible<VPU::NCEConvolutionOp>(op)) {
                        _splitOverHeightEfficencies.insert({op, computeLayerSplitOverHeightEfficency(op)});
                    } else {
                        _splitOverHeightEfficencies.insert({op, 0});
                    }
                    // Check if the operation SOK compatible, otherwise set the efficiency to 0
                    if (isOperationSplitOverKernelCompatible<VPU::NCEConvolutionOp>(op)) {
                        _splitOverKernelEfficencies.insert({op, computeLayerSplitOverKernelEfficency(op)});
                    } else {
                        _splitOverKernelEfficencies.insert({op, 0});
                    }
                    // Assign the most strategy
                    assignMultiClusterStrategy(op);
                })
                .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp op) {
                    // Check if the operation SOH compatible, otherwise set the efficiency to 0
                    if (isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                        _splitOverHeightEfficencies.insert({op, computeLayerSplitOverHeightEfficency(op)});
                    } else {
                        _splitOverHeightEfficencies.insert({op, 0});
                    }
                    // Check if the operation SOK compatible, otherwise set the efficiency to 0
                    if (isOperationSplitOverKernelCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                        _splitOverKernelEfficencies.insert({op, computeLayerSplitOverKernelEfficency(op)});
                    } else {
                        _splitOverKernelEfficencies.insert({op, 0});
                    }
                    // Assign the most strategy
                    assignMultiClusterStrategy(op);
                });
    };

    _func.walk(callback);
}

void StrategyManager::assignMultiClusterStrategy(mlir::Operation* op) {
    llvm::TypeSwitch<mlir::Operation*, void>(op)
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp op) {
                // If operation is neither SOH or SOK compatible, then it has clustering strategy
                if (!isOperationSplitOverHeightCompatible<VPU::NCEConvolutionOp>(op) &&
                    !isOperationSplitOverKernelCompatible<VPU::NCEConvolutionOp>(op)) {
                    op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "Clustering"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'", op->getAttr(multiClusterStrategy),
                               op->getName());
                }
                // Compare the SOH and SOK efficiencies and select SOH if they are equal
                else if (_splitOverHeightEfficencies.find(op)->second >= _splitOverKernelEfficencies.find(op)->second) {
                    // A channel major conv should be have SplitOverHOverLapped strategy
                    if (DimsOrder::fromValue(op.input()) == DimsOrder::NCHW) {
                        op->setAttr(multiClusterStrategy,
                                    mlir::StringAttr::get(op->getContext(), "SplitOverHeightOverLapped"));
                        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                                   op->getAttr(multiClusterStrategy), op->getName());
                    } else {
                        op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "SplitOverHeight"));
                        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                                   op->getAttr(multiClusterStrategy), op->getName());
                    }
                }
                // SOK is more efficient than SOH
                else if (_splitOverHeightEfficencies.find(op)->second < _splitOverKernelEfficencies.find(op)->second) {
                    op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "SplitOverKernel"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'", op->getAttr(multiClusterStrategy),
                               op->getName());
                }
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp op) {
                const auto inputType = op.input().getType().cast<mlir::ShapedType>();
                const auto inputShape = getShape(inputType);
                const auto filterShape = Shape(parseIntArrayAttr<int64_t>(op.rawFilterShapeAttr()));
                const auto IC = inputShape[Dims4D::Act::C];
                const auto IH = inputShape[Dims4D::Act::H];
                const auto IW = inputShape[Dims4D::Act::W];
                const double KY = filterShape[Dims4D::Filter::KY];
                const double KX = filterShape[Dims4D::Filter::KX];
                const double WOC = filterShape[Dims4D::Filter::OC];
                const double inputTensorVolume = IC * IH * IW;  // Need to add precision
                const double weightTensorVolume =
                        WOC * 1 * std::ceil((1 * KY * KX) / 16) * 16;  // Need to add precision

                // If operation is neither SOH or SOK compatible, then it has clustering strategy
                if (!isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op) &&
                    !isOperationSplitOverKernelCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "Clustering"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'", op->getAttr(multiClusterStrategy),
                               op->getName());
                }
                // If the SOH and SOK efficiency values are equal
                // and the operation is both SOH and SOK compatible
                // then select either SOK or SOH based on the lesser
                // amount of data that has to be moved
                else if ((_splitOverHeightEfficencies.find(op)->second ==
                          _splitOverKernelEfficencies.find(op)->second) &&
                         isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op) &&
                         isOperationSplitOverKernelCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    // If the
                    if (_numClusters * inputTensorVolume + weightTensorVolume <
                        inputTensorVolume + (_numClusters * weightTensorVolume)) {
                        op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "SplitOverKernel"));
                        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                                   op->getAttr(multiClusterStrategy), op->getName());
                    } else {
                        op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "SplitOverHeight"));
                        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                                   op->getAttr(multiClusterStrategy), op->getName());
                    }
                } else if ((_splitOverHeightEfficencies.find(op)->second >
                            _splitOverKernelEfficencies.find(op)->second) &&
                           isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "SplitOverHeight"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'", op->getAttr(multiClusterStrategy),
                               op->getName());

                } else if ((_splitOverHeightEfficencies.find(op)->second >
                            _splitOverKernelEfficencies.find(op)->second) &&
                           !isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "Clustering"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'", op->getAttr(multiClusterStrategy),
                               op->getName());
                } else if ((_splitOverHeightEfficencies.find(op)->second <
                            _splitOverKernelEfficencies.find(op)->second) &&
                           isOperationSplitOverKernelCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "SplitOverKernel"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'", op->getAttr(multiClusterStrategy),
                               op->getName());

                } else if ((_splitOverHeightEfficencies.find(op)->second <
                            _splitOverKernelEfficencies.find(op)->second) &&
                           !isOperationSplitOverKernelCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    op->setAttr(multiClusterStrategy, mlir::StringAttr::get(op->getContext(), "Clustering"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'", op->getAttr(multiClusterStrategy),
                               op->getName());
                }
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp op) {
                assignMultiClusterStrategyForEltwiseAndMaxPool<VPU::NCEMaxPoolOp>(op);
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp op) {
                assignMultiClusterStrategyForEltwiseAndMaxPool<VPU::NCEEltwiseOp>(op);
            });
}

mlir::ArrayAttr StrategyManager::getKernelSize(VPU::NCEDepthConvolutionOp& origOp) const {
    const Shape filterShape = origOp.rawFilterShape().hasValue()
                                      ? Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShape().getValue()))
                                      : getShape(origOp.filter()).toValues();
    return getIntArrayAttr(const_cast<mlir::FuncOp&>(_func).getContext(),
                           makeArrayRef({filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]}));
}

mlir::ArrayAttr StrategyManager::getKernelSize(VPU::NCEConvolutionOp& origOp) const {
    const Shape filterShape = origOp.rawFilterShape().hasValue()
                                      ? Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShape().getValue()))
                                      : getShape(origOp.filter()).toValues();
    return getIntArrayAttr(const_cast<mlir::FuncOp&>(_func).getContext(),
                           makeArrayRef({filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]}));
}

mlir::ArrayAttr StrategyManager::getKernelSize(VPU::NCEMaxPoolOp& origOp) const {
    return origOp.kernel_size();
}

mlir::ArrayAttr StrategyManager::getKernelSize(VPU::NCEEltwiseOp& origOp) const {
    return getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1}));
}

mlir::ArrayAttr StrategyManager::getStride(VPU::NCEDepthConvolutionOp& origOp) const {
    return origOp.strides();
}

mlir::ArrayAttr StrategyManager::getStride(VPU::NCEConvolutionOp& origOp) const {
    return origOp.strides();
}

mlir::ArrayAttr StrategyManager::getStride(VPU::NCEMaxPoolOp& origOp) const {
    return origOp.strides();
}

mlir::ArrayAttr StrategyManager::getStride(VPU::NCEEltwiseOp& origOp) const {
    return getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1}));
}

vpux::VPU::PaddingAttr StrategyManager::getPad(VPU::NCEDepthConvolutionOp& origOp) const {
    return origOp.padAttr();
}

vpux::VPU::PaddingAttr StrategyManager::getPad(VPU::NCEConvolutionOp& origOp) const {
    return origOp.padAttr();
}

vpux::VPU::PaddingAttr StrategyManager::getPad(VPU::NCEMaxPoolOp& origOp) const {
    return origOp.padAttr();
}

vpux::VPU::PaddingAttr StrategyManager::getPad(VPU::NCEEltwiseOp& origOp) const {
    return VPU::getPaddingAttr(origOp.getContext(), 0, 0, 0, 0);
}

void StrategyManager::removeStrategyAttribute() {
    _func->walk([](mlir::Operation* op) {
        if (op->hasAttr(multiClusterStrategy)) {
            op->removeAttr(multiClusterStrategy);
        }
    });
}
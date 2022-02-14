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

StrategyManager::StrategyManager(mlir::FuncOp func, Logger log): _log(log), _func(func) {
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto nceOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
    auto dpuOp = nceOp.getSubExecutor(VPU::ExecutorKindAttr::get(module->getContext(), VPU::ExecutorKind::DPU));

    _numClusters = nceOp.count();
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

double getAlignment(double input, size_t unit) {
    return std::ceil(input / unit) * unit;
}

double StrategyManager::calculateSplitOverHeightEfficency(mlir::Operation* op) {
    const auto splitOverHeightFormula = [&](double OH, double OW, double OC) {
        double outputTensorVolume = OC * OH * OW;
        return std::max((outputTensorVolume / _numClusters) /
                                getAlignment((getAlignment(std::ceil(OH / _numClusters), _numClusters) *
                                              getAlignment(OW, _numClusters) * getAlignment(OC, _numChannelAlignment)),
                                             _numDPUPerCluster),
                        (outputTensorVolume / _numClusters) /
                                getAlignment((getAlignment(std::ceil(OH / _numClusters), _numChannelAlignment) *
                                              getAlignment(OW, 1) * getAlignment(OC, _numChannelAlignment)),
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
                const auto KY = filterShape[Dims4D::Filter::KY];

                const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
                const double outputTensorVolume = OC * OH * OW;
                double efficency = 0;

                // Different efficency formula required for CM Conv and ZM Conv
                // CM Conv
                if (DimsOrder::fromValue(origOp.input()) == DimsOrder::NCHW) {
                    auto efficiencyConstant = channelMajorEfficiencyTable()[KY][strides[0]];
                    efficency = efficiencyConstant *
                                std::max(outputTensorVolume /
                                                 (getAlignment(OH, _numChannelAlignment) * getAlignment(OH, _numDPU) *
                                                  getAlignment(OC, _numChannelAlignment)),
                                         outputTensorVolume / (getAlignment(OH, _numChannelAlignment * _numClusters) *
                                                               getAlignment(OW, _numDPUPerCluster) *
                                                               getAlignment(OC, _numChannelAlignment)));
                } else {  // ZM Conv
                    efficency = splitOverHeightFormula(OH, OW, OC);
                }
                _log.trace("The SOH efficiency for the convolution is {0}", efficency);
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
                double efficency = efficiencyConstant * splitOverHeightFormula(OH, OW, OC);
                //_log.trace("The SOH efficiency for the group convolution is {0}", efficency);
                return efficency;
            });
}

double StrategyManager::calculateSplitOverKernelEfficency(mlir::Operation* op) {
    const auto splitOverKernelFormula = [&](double OH, double OW, double OC) {
        double outputTensorVolume = OC * OH * OW;
        return std::max((outputTensorVolume / _numClusters) /
                                getAlignment((getAlignment(OH, _numClusters) * getAlignment(OW, _numClusters) *
                                              getAlignment(std::ceil(OC / _numClusters), _numChannelAlignment)),
                                             _numDPUPerCluster),
                        (outputTensorVolume / _numClusters) /
                                getAlignment((getAlignment(OH, _numChannelAlignment) * getAlignment(OW, 1) *
                                              getAlignment(std::ceil(OC / _numClusters), _numChannelAlignment)),
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

// Computes the multi-cluster efficiency value for operation
// If it is not compatible with the multi-cluster strategy the efficiency is 0
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
                    // Check if the operation SOH compatible, otherwise the efficiency is 0
                    if (isOperationSplitOverHeightCompatible<VPU::NCEConvolutionOp>(op)) {
                        _splitOverHeightEfficencies.insert({op, calculateSplitOverHeightEfficency(op)});
                    } else {
                        _splitOverHeightEfficencies.insert({op, 0});
                    }
                    // Check if the operation SOK compatible, otherwise the efficiency is 0
                    if (isOperationSplitOverKernelCompatible<VPU::NCEConvolutionOp>(op)) {
                        _splitOverKernelEfficencies.insert({op, calculateSplitOverKernelEfficency(op)});
                    } else {
                        _splitOverKernelEfficencies.insert({op, 0});
                    }
                    // Assign the most strategy
                    assignMultiClusterStrategy(op);
                })
                .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp op) {
                    // Check if the operation SOH compatible, otherwise the efficiency is 0
                    if (isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                        _splitOverHeightEfficencies.insert({op, calculateSplitOverHeightEfficency(op)});
                    } else {
                        _splitOverHeightEfficencies.insert({op, 0});
                    }
                    // Check if the operation SOK compatible, otherwise the efficiency is 0
                    if (isOperationSplitOverKernelCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                        _splitOverKernelEfficencies.insert({op, calculateSplitOverKernelEfficency(op)});
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

VPU::DistributionMode StrategyManager::getActivationTensorDistributionMode(const llvm::StringRef multiClusterStrategy) {
    if (multiClusterStrategy == splitOverHeightOverLappedStrategy) {
        return VPU::DistributionMode::overlapped;
    } else if (multiClusterStrategy == splitOverHeightStrategy) {
        return VPU::DistributionMode::segmented;
    } else if (multiClusterStrategy == splitOverKernelStrategy) {
        return VPU::DistributionMode::multicasted;
    } else {
        VPUX_THROW("Operation was not assigned a valid multi-cluster strategy, unable to determine a distribution mode "
                   "for the activation tensor");
    }
}

VPU::DistributionMode StrategyManager::getWeightsTensorDistributionMode(const llvm::StringRef multiClusterStrategy) {
    if (multiClusterStrategy == splitOverHeightOverLappedStrategy) {
        return VPU::DistributionMode::multicasted;
    } else if (multiClusterStrategy == splitOverHeightStrategy) {
        return VPU::DistributionMode::multicasted;
    } else if (multiClusterStrategy == splitOverKernelStrategy) {
        return VPU::DistributionMode::segmented;
    } else {
        VPUX_THROW("Operation was not assigned a valid multi-cluster strategy, unable to determine a distribution mode "
                   "for the weights tensor");
    }
}

llvm::ArrayRef<int32_t> StrategyManager::getActivationTensorNumTiles(const llvm::StringRef multiClusterStrategy) {
    if (multiClusterStrategy == splitOverHeightOverLappedStrategy) {
        return ArrayRef<int32_t>{1, 1, 4, 1};  // Use num clusters from IR
    } else if (multiClusterStrategy == splitOverHeightStrategy) {
        return ArrayRef<int32_t>{1, 1, 4, 1};  // Use num clusters from IR
    } else if (multiClusterStrategy == splitOverKernelStrategy) {
        return ArrayRef<int32_t>{1, 1, 1, 1};  // Use num clusters from IR
    } else {
        VPUX_THROW("Operation was not assigned a valid multi-cluster strategy, unable to determine a number of tiles "
                   "for the activation tensor");
    }
}

llvm::ArrayRef<int32_t> StrategyManager::getWeightsTensorNumTiles(const llvm::StringRef multiClusterStrategy) {
    if (multiClusterStrategy == splitOverHeightOverLappedStrategy) {
        return ArrayRef<int32_t>{1, 1, 1, 1};  // Use num clusters from IR
    } else if (multiClusterStrategy == splitOverHeightStrategy) {
        return ArrayRef<int32_t>{1, 1, 1, 1};  // Use num clusters from IR
    } else if (multiClusterStrategy == splitOverKernelStrategy) {
        return ArrayRef<int32_t>{1, 1, 4, 1};  // Use num clusters from IR
    } else {
        VPUX_THROW("Operation was not assigned a valid multi-cluster strategy, unable to determine a number of tiles "
                   "for the weights tensor");
    }
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

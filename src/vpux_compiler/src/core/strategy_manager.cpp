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

StrategyManager::StrategyManager(mlir::FuncOp func, size_t numClusters, Logger log)
        : _log(log), _numClusters(numClusters), _func(func) {
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
                const auto inputType = origOp.input().getType().cast<mlir::ShapedType>();
                const auto outputShape = getShape(outputType);
                const auto inputShape = getShape(inputType);
                const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
                const auto IC = inputShape[Dims4D::Act::C];
                const double OC = outputShape[Dims4D::Act::C];
                const double OH = outputShape[Dims4D::Act::H];
                const double OW = outputShape[Dims4D::Act::W];
                const auto KY = filterShape[Dims4D::Filter::KY];

                const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
                const double outputTensorVolume = OC * OH * OW;
                double efficency = 0;

                // Different efficency formula required for CM Conv and ZM Conv
                // CM Conv
                if (IC == 3) {
                    auto efficiencyConstant = channelMajorEfficiencyTable()[KY][strides[0]];
                    efficency = efficiencyConstant *
                                std::max(outputTensorVolume /
                                                 (getAlignment(OH, _numChannelAlignment) * getAlignment(OH, _numDPU) *
                                                  getAlignment(OC, _numChannelAlignment)),
                                         outputTensorVolume / (getAlignment(OH, _numChannelAlignment * _numClusters) *
                                                               getAlignment(OW, _numDPUPerCluster) *
                                                               getAlignment(OC, _numChannelAlignment)));
                    // ZM Conv
                } else {
                    efficency = splitOverHeightFormula(OH, OW, OC);
                }
                _log.trace("The SOH efficiency for the convolution is {0}", efficency);
                return efficency;
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp origOp) {
                return 1;
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp origOp) {
                return 1;
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp origOp) {
                const auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
                const auto inputType = origOp.input().getType().cast<mlir::ShapedType>();
                const auto outputShape = getShape(outputType);
                const auto inputShape = getShape(inputType);
                const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
                const double OC = outputShape[Dims4D::Act::C];
                const double OH = outputShape[Dims4D::Act::H];
                const double OW = outputShape[Dims4D::Act::W];
                const auto KY = filterShape[Dims4D::Filter::KY];
                const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
                const double outputTensorVolume = OC * OH * OW;

                auto efficiencyConstant = depthwiseEfficiencyTable()[KY][strides[0]];
                double efficency = efficiencyConstant * splitOverHeightFormula(OH, OW, OC);
                _log.trace("The SOH efficiency for the group convolution is {0}", efficency);
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
                const auto inputType = origOp.input().getType().cast<mlir::ShapedType>();
                const auto outputShape = getShape(outputType);
                const auto inputShape = getShape(inputType);
                const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
                const double OC = outputShape[Dims4D::Act::C];
                const double OH = outputShape[Dims4D::Act::H];
                const double OW = outputShape[Dims4D::Act::W];
                const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
                double efficency = splitOverKernelFormula(OH, OW, OC);
                _log.trace("The SOK efficiency for the convolution is {0}", efficency);
                return efficency;
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp origOp) {
                return 1;
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp origOp) {
                return 1;
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp origOp) {
                const auto outputType = origOp.output().getType().cast<mlir::ShapedType>();
                const auto inputType = origOp.input().getType().cast<mlir::ShapedType>();
                const auto outputShape = getShape(outputType);
                const auto inputShape = getShape(inputType);
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
                .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp op) {

                })
                .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp op) {
                    assignMultiClusterStrategy(op);
                })
                .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp op) {
                    // Is operation SOH compatible
                    if (isOperationSplitOverHeightCompatible<VPU::NCEConvolutionOp>(op)) {
                        _splitOverHeightEfficencies.insert({op, calculateSplitOverHeightEfficency(op)});
                    } else {
                        _splitOverHeightEfficencies.insert({op, 0});
                    }
                    // Is operation SOK compatible
                    if (isOperationSplitOverKernelCompatible<VPU::NCEConvolutionOp>(op)) {
                        _splitOverKernelEfficencies.insert({op, calculateSplitOverKernelEfficency(op)});
                    } else {
                        _splitOverKernelEfficencies.insert({op, 0});
                    }
                    // Assign the most strategy
                    assignMultiClusterStrategy(op);
                })
                .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp op) {
                    // Is operation SOH compatible
                    if (isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                        _splitOverHeightEfficencies.insert({op, calculateSplitOverHeightEfficency(op)});
                    } else {
                        _splitOverHeightEfficencies.insert({op, 0});
                    }
                    // Is operation SOK compatible
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
                // If operation is neither SOH or SOK compatible, then it has to be Clustering
                if (!isOperationSplitOverHeightCompatible<VPU::NCEConvolutionOp>(op) &&
                    !isOperationSplitOverKernelCompatible<VPU::NCEConvolutionOp>(op)) {
                    op->setAttr(multiClusterStrategyAttrName, mlir::StringAttr::get(op->getContext(), "Clustering"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                               op->getAttr(multiClusterStrategyAttrName), op->getName());
                }
                // Compare the SOK and SOK efficiencies
                // Select SOH if they are equal
                else if (_splitOverHeightEfficencies.find(op)->second >= _splitOverKernelEfficencies.find(op)->second) {
                    op->setAttr(multiClusterStrategyAttrName, mlir::StringAttr::get(op->getContext(), "SplitOverH"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                               op->getAttr(multiClusterStrategyAttrName), op->getName());

                }
                // SOK is more efficient than SOH
                else if (_splitOverHeightEfficencies.find(op)->second < _splitOverKernelEfficencies.find(op)->second) {
                    op->setAttr(multiClusterStrategyAttrName, mlir::StringAttr::get(op->getContext(), "SplitOverK"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                               op->getAttr(multiClusterStrategyAttrName), op->getName());
                }
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp op) {
                const auto outputType = op.output().getType().cast<mlir::ShapedType>();
                const auto inputType = op.input().getType().cast<mlir::ShapedType>();
                const auto outputShape = getShape(outputType);
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

                // If operation is neither SOH or SOK compatible, then it has to be Clustering
                if (!isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op) &&
                    !isOperationSplitOverKernelCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    op->setAttr(multiClusterStrategyAttrName, mlir::StringAttr::get(op->getContext(), "Clustering"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                               op->getAttr(multiClusterStrategyAttrName), op->getName());
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
                        op->setAttr(multiClusterStrategyAttrName,
                                    mlir::StringAttr::get(op->getContext(), "SplitOverK"));
                        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                                   op->getAttr(multiClusterStrategyAttrName), op->getName());
                    } else {
                        op->setAttr(multiClusterStrategyAttrName,
                                    mlir::StringAttr::get(op->getContext(), "SplitOverH"));
                        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                                   op->getAttr(multiClusterStrategyAttrName), op->getName());
                    }
                } else if ((_splitOverHeightEfficencies.find(op)->second >
                            _splitOverKernelEfficencies.find(op)->second) &&
                           isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    op->setAttr(multiClusterStrategyAttrName, mlir::StringAttr::get(op->getContext(), "SplitOverH"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                               op->getAttr(multiClusterStrategyAttrName), op->getName());

                } else if ((_splitOverHeightEfficencies.find(op)->second >
                            _splitOverKernelEfficencies.find(op)->second) &&
                           !isOperationSplitOverHeightCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    op->setAttr(multiClusterStrategyAttrName, mlir::StringAttr::get(op->getContext(), "Clustering"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                               op->getAttr(multiClusterStrategyAttrName), op->getName());
                } else if ((_splitOverHeightEfficencies.find(op)->second <
                            _splitOverKernelEfficencies.find(op)->second) &&
                           isOperationSplitOverKernelCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    op->setAttr(multiClusterStrategyAttrName, mlir::StringAttr::get(op->getContext(), "SplitOverK"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                               op->getAttr(multiClusterStrategyAttrName), op->getName());

                } else if ((_splitOverHeightEfficencies.find(op)->second <
                            _splitOverKernelEfficencies.find(op)->second) &&
                           !isOperationSplitOverKernelCompatible<VPU::NCEDepthConvolutionOp>(op)) {
                    op->setAttr(multiClusterStrategyAttrName, mlir::StringAttr::get(op->getContext(), "Clustering"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                               op->getAttr(multiClusterStrategyAttrName), op->getName());
                }
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp op) {
                assignMultiClusterStrategyForEltwise<VPU::NCEEltwiseOp>(op);
            });
}
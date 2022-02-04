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
                    const auto inputType = op.input().getType().cast<mlir::ShapedType>();
                    const auto inputShape = getShape(inputType);
                    const auto IC = inputShape[Dims4D::Act::C];

                    // Channel Major Conv should be SplitOverHOverLapped
                    if (IC <= 3) {
                        op->setAttr(multiClusterStrategyAttrName,
                                    mlir::StringAttr::get(op->getContext(), "SplitOverHOverLapped"));
                        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                                   op->getAttr(multiClusterStrategyAttrName), op->getName());
                    } else {
                        op->setAttr(multiClusterStrategyAttrName,
                                    mlir::StringAttr::get(op->getContext(), "SplitOverH"));
                        _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                                   op->getAttr(multiClusterStrategyAttrName), op->getName());
                    }
                }
                // SOK is more efficient than SOH
                else if (_splitOverHeightEfficencies.find(op)->second < _splitOverKernelEfficencies.find(op)->second) {
                    op->setAttr(multiClusterStrategyAttrName, mlir::StringAttr::get(op->getContext(), "SplitOverK"));
                    _log.trace("Assign multi-cluster strategy '{0}' to layer '{1}'",
                               op->getAttr(multiClusterStrategyAttrName), op->getName());
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

void StrategyManager::insertCopyOpForDistributedTensor() {
    const auto callback = [&](mlir::Operation* origOp) {
        llvm::TypeSwitch<mlir::Operation*, void>(origOp)
                .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp) {})
                .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp) {})
                .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp origOp) {
                    _log.trace("Got operation {0}", origOp);
                    if (origOp->getParentOfType<VPU::NCEClusterTilingOp>()) {
                        _log.trace("The operation is already wrapped");
                        return mlir::failure();
                    }

                    // Retrieve the strategy
                    const auto strategy =
                            origOp->getAttr(multiClusterStrategyAttrName).cast<mlir::StringAttr>().getValue();

                    // Create the Copy op for the distributed tensor for SOH OverLapped Operation
                    if (strategy == splitOverHeightOverLappedStrategyAttrName) {
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////
                        //// Create the copy operation for the distributed activation tensor for SOH OverLappe Operation
                        //////
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////

                        // Step 1: Create DistributedTensorAttr fields
                        // Specify the distribution mode of the tensor  overlapped,duplicated,segmented, multicasted,
                        const auto activationTensorDistributionModeAttr = vpux::VPU::DistributionModeAttr::get(
                                origOp.getContext(), vpux::VPU::DistributionMode::overlapped);

                        // Specify the number of tiles (clusters)
                        const auto numTiles =
                                getIntArrayAttr(origOp.getContext(), makeArrayRef({1, 1, (int)_numClusters, 1}));

                        // Specify the kernel
                        const auto filterShape = getShape(origOp.filter());
                        const auto kernel = getIntArrayAttr(
                                origOp.getContext(),
                                makeArrayRef({filterShape[Dims4D::Filter::KY],
                                              filterShape[Dims4D::Filter::KX]}));  // TODO: Is this the correct order of
                                                                                   // dims?

                        // Step 2: Create DistributedTensorAttr
                        auto activationTensorDistributedTensorAttr = vpux::VPU::DistributedTensorAttr::get(
                                activationTensorDistributionModeAttr, numTiles, kernel, origOp.strides(),
                                origOp.padAttr(),
                                origOp.getContext());  // TODO: Use the padding from origOp?

                        // Step 3: Create DistributedTensorType fields
                        const auto inputType = origOp.input().getType().cast<mlir::ShapedType>();

                        // Specify the inputShape
                        const auto inputShape = getShape(inputType);
                        SmallVector<int64_t> inShape{
                                inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C], inputShape[Dims4D::Act::H],
                                inputShape[Dims4D::Act::W]};  // TODO: Is this the correct order of dims?

                        // Specify the memSpace
                        const auto memSpace = mlir::SymbolRefAttr::get(
                                VPU::MemoryKindAttr::get(origOp.getContext(), VPU::MemoryKind::CMX_NN));

                        // Specify the order
                        const auto order = mlir::AffineMapAttr::get(
                                DimsOrder::fromType(origOp.input().getType().cast<mlir::ShapedType>())
                                        .toAffineMap(origOp.getContext()));

                        // Step 4: Create DistributedTensorType
                        const auto activationTensorDistributedTensorType = vpux::VPU::DistributedTensorType::get(
                                origOp.getContext(), inShape, origOp.input().getType().cast<mlir::ShapedType>(), order,
                                memSpace, activationTensorDistributedTensorAttr);

                        _log.trace("Wrap copy operation for activation into NCEClusterTilingOp");

                        // Step 5: Create IE::Copy Op
                        mlir::OpBuilder builder(_func.getBody());
                        const auto activationTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                                     mlir::ValueRange newOperands) {
                            const auto memSpace = IndexedSymbolAttr::get(builder.getContext(),
                                                                         stringifyEnum(VPU::MemoryKind::CMX_NN));
                            auto activationTensorDistributedCopyOp =
                                    builder.create<IE::CopyOp>(origOp->getLoc(), newOperands[0], memSpace);
                            builder.create<VPU::YieldOp>(loc, activationTensorDistributedCopyOp->getResults());
                        };

                        // Step 6: Wrap the IE::Copy Op in NCEClusterTiling
                        builder.create<VPU::NCEClusterTilingOp>(origOp->getLoc(), activationTensorDistributedTensorType,
                                                                origOp->getOperands(), activationTensorBodyBuilder);

                        /////////////////////////////////////////////////////////////////////////////////////////////////////////
                        //// Create the copy operation for the multicasted weights for SOH OverLapped Operation
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////

                        const auto weightsTensorDistributionModeAttr = vpux::VPU::DistributionModeAttr::get(
                                origOp.getContext(), vpux::VPU::DistributionMode::multicasted);

                        // Step 2: Create DistributedTensorAttr
                        auto weightsTensorDistributedTensorAttr = vpux::VPU::DistributedTensorAttr::get(
                                weightsTensorDistributionModeAttr, numTiles, kernel, origOp.strides(), origOp.padAttr(),
                                origOp.getContext());  // TODO: Use the padding from origOp?

                        // Use the same fields from activation tensor  for step3 & 4

                        // Step 4: Create DistributedTensorType
                        const auto weightsTensorDistributedTensorType = vpux::VPU::DistributedTensorType::get(
                                origOp.getContext(), inShape, origOp.input().getType().cast<mlir::ShapedType>(), order,
                                memSpace, weightsTensorDistributedTensorAttr);

                        _log.trace("Wrap copy operation for weights into NCEClusterTilingOp");

                        // Step 5: Create IE::Copy Op
                        const auto weightsTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                                  mlir::ValueRange newOperands) {
                            const auto memSpace = IndexedSymbolAttr::get(builder.getContext(),
                                                                         stringifyEnum(VPU::MemoryKind::CMX_NN));
                            auto weightsTensorDistributedCopyOp =
                                    builder.create<IE::CopyOp>(origOp->getLoc(), newOperands[0], memSpace);
                            builder.create<VPU::YieldOp>(loc, weightsTensorDistributedCopyOp->getResults());
                        };

                        // Step 6: Wrap the IE::Copy Op in NCEClusterTiling
                        builder.create<VPU::NCEClusterTilingOp>(origOp->getLoc(), weightsTensorDistributedTensorType,
                                                                origOp.input(), weightsTensorBodyBuilder);
                    }
                    return mlir::success();
                })

                .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp) {

                });
    };

    _func.walk(callback);
}
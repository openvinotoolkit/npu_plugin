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

#include "vpux/compiler/dialect/IE/strategy_manager.hpp"
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

StrategyManager::StrategyManager(mlir::FuncOp func, size_t numClusters, Logger log)
        : _log(log), _numClusters(numClusters), _func(func) {
}

// This channel major efficiency table is from the ArchBench tool
std::map<double, std::map<int64_t,int64_t>> StrategyManager::channelMajorEfficiencyTable() {
    return { {
       { 0.165,  { {1, 3} } },
       { 0.128,  { {2, 3} } },
       { 0.128,  { {4, 3} } },
       { 0.165,  { {6, 3} } },
       { 0.483,  { {1, 5} } },
       { 0.241,  { {2, 5} } },
       { 0.132,  { {4, 5} } },
       { 0.483,  { {6, 5} } },
       { 0.6,    { {1, 7} } },
       { 0.2965, { {2, 7} } },
       { 0.15,   { {4, 7} } },
       { 0.0395, { {6, 7} } },
       { 0.8008, { {1, 9} } },
       { 0.4687, { {2, 9} } },
       { 0.2266, { {4, 9} } },
       { 0.8008, { {6, 9} } },
       { 0.9023, { {1, 11} } },
       { 0.4687, { {2, 11} } },
       { 0.2366, { {4, 11} } },
       { 0.9023, { {6, 11} } },
     } };
}

size_t StrategyManager::calculateSplitOverHeightEfficency(mlir::Operation* op) {
    return llvm::TypeSwitch<mlir::Operation*, size_t>(op)
            .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp origOp) {
                const auto outputType = op->getResult(0).getType().cast<mlir::ShapedType>();
                const auto inputType = op->getOperand(0).getType().cast<mlir::ShapedType>();
                const auto weightsType = op->getOperand(1).getType().cast<mlir::ShapedType>();
                const auto outputShape = getShape(outputType);
                const auto inputShape = getShape(inputType);
                const auto weightsShape = getShape(weightsType);
                const auto IC = inputShape[Dims4D::Act::C];
                const auto OC = outputShape[Dims4D::Act::C];
                const auto OH = outputShape[Dims4D::Act::H];
                const auto OW = outputShape[Dims4D::Act::W];
                const auto KY = weightsShape[Dims4D::Filter::KY];

                const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
                const double outputTensorVolume = OC * OH * OW;
                double efficency = 0;

                // Different efficency formula required for CM Conv and ZM Conv
                if (IC == 3) {
                    auto constant = channelMajorEfficiencyTable()[strides[0]][KY];
                    efficency = constant *
                                std::max(outputTensorVolume / (16.0 * std::ceil(OH / 16.0) * 20.0 *
                                                               std::ceil(OW / 20.0) * 16.0 * std::ceil(OC / 16.0)),
                                         outputTensorVolume / (64.0 * std::ceil(OH / 64.0) * 5.0 * std::ceil(OW / 5.0) *
                                                               16.0 * std::ceil(OC / 16.0)));
                } else {
                }
                return efficency;
            })
            .Case<IE::MaxPoolOp>([&](IE::MaxPoolOp origOp) {
                return 1;
            })
            .Case<IE::AddOp>([&](IE::AddOp origOp) {
                return 1;
            })
            .Case<IE::MultiplyOp>([&](IE::MultiplyOp origOp) {
                return 1;
            })
            .Case<IE::SubtractOp>([&](IE::SubtractOp origOp) {
                return 1;
            })
            .Case<IE::AndOp>([&](IE::AndOp origOp) {
                return 1;
            })
            .Case<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp origOp) {
                const auto outputType = op->getResult(0).getType().cast<mlir::ShapedType>();
                const auto outputShape = getShape(outputType);
                const auto OC = outputShape[Dims4D::Act::C];
                const auto OH = outputShape[Dims4D::Act::H];
                const auto OW = outputShape[Dims4D::Act::W];
                const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());

                double efficency = 0;
                efficency = 0.483 * std::max(((OH / 4.0 * OW * OC) /
                                              (5.0 * std::ceil((4.0 * std::ceil(std::ceil(OH / 4.0) / 4.0) * 4 *
                                                                std::ceil(OW / 4.0) * 16.0 * std::ceil(OC / 16)) /
                                                               5))),
                                             ((OH / 4.0 * OW * OC) /
                                              (5.0 * std::ceil((16.0 * std::ceil(std::ceil(OH / 4.0) / 16.0) * 1.0 *
                                                                std::ceil(OW / 1.0) * 16.0 * std::ceil(OC / 16.0)) /
                                                               5.0))));
                return efficency;
            })
            .Default([](mlir::Operation* unknownOp) {
                VPUX_THROW("Operation is not supported by the NCE");
                return 0;
            });
}

size_t StrategyManager::calculateSplitOverKernelEfficency(mlir::Operation* op) {
    return llvm::TypeSwitch<mlir::Operation*, size_t>(op)
            .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp origOp) {
                const auto outputType = op->getResult(0).getType().cast<mlir::ShapedType>();
                const auto outputShape = getShape(outputType);
                const auto OC = outputShape[Dims4D::Act::C];
                const auto OH = outputShape[Dims4D::Act::H];
                const auto OW = outputShape[Dims4D::Act::W];
                const double outputTensorVolume = OC * OH * OW;
                const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
                const double efficency =
                        0.183594 * std::max(outputTensorVolume / (16.0 * std::ceil(OH / 16.0) * 20.0 *
                                                                  std::ceil(OW / 20.0) * 16.0 * std::ceil(OC / 16.0)),
                                            outputTensorVolume / (64.0 * std::ceil(OH / 64.0) * 5.0 *
                                                                  std::ceil(OW / 5.0) * 16.0 * std::ceil(OC / 16.0)));
                return efficency;
            })
            .Case<IE::MaxPoolOp>([&](IE::MaxPoolOp origOp) {
                return 1;
            })
            .Case<IE::AddOp>([&](IE::AddOp origOp) {
                return 1;
            })
            .Case<IE::MultiplyOp>([&](IE::MultiplyOp origOp) {
                return 1;
            })
            .Case<IE::SubtractOp>([&](IE::SubtractOp origOp) {
                return 1;
            })
            .Case<IE::AndOp>([&](IE::AndOp origOp) {
                return 1;
            })
            .Case<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp origOp) {
                const auto outputType = op->getResult(0).getType().cast<mlir::ShapedType>();
                const auto outputShape = getShape(outputType);
                const auto OC = outputShape[Dims4D::Act::C];
                const auto OH = outputShape[Dims4D::Act::H];
                const auto OW = outputShape[Dims4D::Act::W];

                double efficency =
                        0.483 * std::max(((OH / 4.0 * OW * OC) /
                                          (5.0 * std::ceil((4.0 * std::ceil(std::ceil(OH / 4.0) / 4.0) * 4 *
                                                            std::ceil(OW / 4.0) * 16.0 * std::ceil(OC / 16)) /
                                                           5))),
                                         ((OH / 4.0 * OW * OC) /
                                          (5.0 * std::ceil((16.0 * std::ceil(std::ceil(OH / 4.0) / 16.0) * 1.0 *
                                                            std::ceil(OW / 1.0) * 16.0 * std::ceil(OC / 16.0)) /
                                                           5.0))));

                return efficency;
            })
            .Default([](mlir::Operation* unknownOp) {
                VPUX_THROW("Operation is not supported by the NCE");
                return 0;
            });
}

void StrategyManager::computeOptimalMultiClusterStrategy() {
    const auto callback = [&](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<IE::MaxPoolOp>([&](IE::MaxPoolOp op) {

                })
                .Case<IE::AvgPoolOp>([&](IE::AvgPoolOp op) {})
                .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp op) {
                    if (isOperationSplitOverHeightCompatible<IE::ConvolutionOp>(op)) {
                        _splitOverHeightEfficencies.insert({op, calculateSplitOverHeightEfficency(op)});
                    }
                })
                .Case<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp op) {
                    // Is operation SOH compatible
                    if (isOperationSplitOverHeightCompatible<IE::GroupConvolutionOp>(op)) {
                        _splitOverHeightEfficencies.insert({op, calculateSplitOverHeightEfficency(op)});
                    }
                    // Is operation SOK compatible
                    if (isOperationSplitOverKernelCompatible<IE::GroupConvolutionOp>(op)) {
                        _splitOverKernelEfficencies.insert({op, calculateSplitOverKernelEfficency(op)});
                    }
                });
    };

    _func.walk(callback);
}
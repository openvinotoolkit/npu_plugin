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

StrategyManager::StrategyManager(mlir::FuncOp func, Logger log): _log(log), _func(func) {
}

size_t StrategyManager::calculateSplitOverHeightEfficency(mlir::Operation* op) {
    return llvm::TypeSwitch<mlir::Operation*, size_t>(op)
            .Case<IE::ConvolutionOp>([&](IE::ConvolutionOp origOp) {
                const auto outputType = op->getResult(0).getType().cast<mlir::ShapedType>();
                const auto outputShape = getShape(outputType);
                const auto OC = outputShape[Dims4D::Act::C];
                const auto OH = outputShape[Dims4D::Act::H];
                const auto OW = outputShape[Dims4D::Act::W];

                auto efficency = 0.183594 *
                                 std::max((OC * OH * OW) / (16 * std::ceil(OH / 16) * 20 * std::ceil(OW / 20) * 16 *
                                                            std::ceil(OC / 16)),
                                          (64 * std::ceil(OH / 64) * 5 * std::ceil(OW / 5) * 16 * std::ceil(OC / 16)));
                return 1;
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
                return 1;
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
                    }
                })
                .Case<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp op) {});
    };

    _func.walk(callback);
}
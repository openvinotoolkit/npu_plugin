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
#include "vpux/compiler/dialect/VPU/distributed_tensor_utils.hpp"
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
namespace VPU {

//
// StrategyManager
//

class StrategyManager final {
public:
    explicit StrategyManager(mlir::FuncOp func, Logger log, mlir::MLIRContext* ctx);

public:
    void assignMultiClusterStrategy();

private:
    template <class ConcreteOp>
    bool isOperationSplitOverHeightCompatible(ConcreteOp origOp) const;
    template <class ConcreteOp>
    bool doesSplitOverHeightLayerFitIntoCMX(ConcreteOp origOp) const;

    int32_t _numClusters;
    const long int _minimumHeightForSOH = 20;
    const size_t _numChannelAlignment = 16;
    mlir::FuncOp _func;
    Logger _log;
    mlir::MLIRContext* _ctx;
};

// An operation is SOH compitable if it has an output height of at least 20
// The reason is because the output tensor in each cluster will only have a
// height of 5 (20/4, assuming 4 cluster compilation).
// There are 5 DPUs in a cluster so each DPU will compute at least one output line
template <class ConcreteOp>
bool StrategyManager::isOperationSplitOverHeightCompatible(ConcreteOp origOp) const {
    const auto outputShape = getShape(origOp.output());
    const auto OH = outputShape[Dims4D::Act::H];
    return OH >= _minimumHeightForSOH;
}

template <class ConcreteOp>
bool StrategyManager::doesSplitOverHeightLayerFitIntoCMX(ConcreteOp origOp) const {
    auto activationTensorDistributionMode = DistributionMode::SEGMENTED;
    auto activationTensorNumTiles = getIntArrayAttr(_ctx, makeArrayRef({1, 1, _numClusters, 1}));
    auto weightsTensorDistributionMode = DistributionMode::MULTICASTED;
    auto weightTensorNumTiles = getIntArrayAttr(_ctx, makeArrayRef({1, 1, 1, 1}));
    auto distributedOutputTensorType =
            createDistributedOutputTensorType(origOp, activationTensorDistributionMode, activationTensorNumTiles);
    auto outputShape = getShape(origOp.output());
    auto OC = outputShape[Dims4D::Act::C];
    Byte totalMemorySize(0);
    int64_t activationWindowSize = 0;

    if (auto convolutionOp = mlir::dyn_cast<NCEConvolutionOp>(origOp.getOperation())) {
        auto distributedActivationTensorType = createDistributedInputTensorType(
                convolutionOp, convolutionOp.input(), activationTensorDistributionMode, activationTensorNumTiles);
        auto distributeddWeightsTensorType = createDistributedInputTensorType(
                convolutionOp, convolutionOp.filter(), weightsTensorDistributionMode, weightTensorNumTiles);
        auto weightsTable = NCEInvariant::getWeightsTableSize(OC);

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
            activationWindowSize =
                    NCESparsity::getActivationWindowSize(NCESparsity::Mode::CM_CONV, kernelSize, SX, elemType, IC);
        }

        totalMemorySize += distributedActivationTensorType.getTotalAllocSize() +
                           distributeddWeightsTensorType.getTotalAllocSize() +
                           distributedOutputTensorType.getTotalAllocSize() + weightsTable +
                           activationWindowSize * 1_Byte;
    } else if (auto depthwiseConvolutionOp = mlir::dyn_cast<NCEDepthConvolutionOp>(origOp.getOperation())) {
        auto distributedActivationTensorType =
                createDistributedInputTensorType(depthwiseConvolutionOp, depthwiseConvolutionOp.input(),
                                                 activationTensorDistributionMode, activationTensorNumTiles);
        auto distributeddWeightsTensorType =
                createDistributedInputTensorType(depthwiseConvolutionOp, depthwiseConvolutionOp.filter(),
                                                 weightsTensorDistributionMode, weightTensorNumTiles);
        auto weightsTable = NCEInvariant::getWeightsTableSize(OC);

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
        activationWindowSize =
                NCESparsity::getActivationWindowSize(NCESparsity::Mode::DW_CONV, kernelSize, SX, elemType, IC);

        totalMemorySize += distributedActivationTensorType.getTotalAllocSize() +
                           distributeddWeightsTensorType.getTotalAllocSize() +
                           distributedOutputTensorType.getTotalAllocSize() + weightsTable +
                           activationWindowSize * 1_Byte;
    } else if (auto maxPoolOp = mlir::dyn_cast<NCEMaxPoolOp>(origOp.getOperation())) {
        auto distributedActivationTensorType = createDistributedInputTensorType(
                maxPoolOp, maxPoolOp.input(), activationTensorDistributionMode, activationTensorNumTiles);
        auto weightsTable = NCEInvariant::getWeightsTableSize(OC);

        const auto inputShape = getShape(maxPoolOp.input());
        const auto IC = inputShape[Dims4D::Act::C];
        const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(maxPoolOp.kernel_size()));
        const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(maxPoolOp.strides()));
        const auto SX = kernelStrides[Dims4D::Strides::X];
        auto elemType = maxPoolOp.input().getType().template cast<mlir::ShapedType>().getElementType();
        activationWindowSize =
                NCESparsity::getActivationWindowSize(NCESparsity::Mode::POOL, kernelSize, SX, elemType, IC);

        totalMemorySize += distributedActivationTensorType.getTotalAllocSize() +
                           distributedOutputTensorType.getTotalAllocSize() + weightsTable +
                           activationWindowSize * 1_Byte;
    } else if (auto eltwiseOp = mlir::dyn_cast<NCEEltwiseOp>(origOp.getOperation())) {
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

    return totalMemorySize <= getTotalCMXSize(origOp.getOperation());
}
}  // namespace VPU
}  // namespace vpux

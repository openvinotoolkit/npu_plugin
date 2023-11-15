//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/layer_vpunn_cost.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/core/cost_model_utils.hpp"

using namespace vpux;
using namespace VPU;

StrategyCost LayerVPUNNCost::getStrategyCost(mlir::Operation* operation, const VPUNNCostParameters& parameters) const {
    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(operation)) {
        return getNCELayerCost(nceOp, parameters);
    } else if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(operation)) {
        return getSWLayerCost(swOp, parameters);
    } else if (VPU::isPureViewOp(operation)) {
        return 0.0;
    } else {
        _log.trace("Unsupported op type {0} at {1}", operation->getName(), operation->getLoc());
        return getSimpleLayerCost(operation, parameters);
    }
}

size_t LayerVPUNNCost::getNumClusterCorrectionSize(VPU::MultiClusterStrategy strategy) const {
    return strategy != MultiClusterStrategy::Clustering ? _numClusters : 1;
}

StrategyCost LayerVPUNNCost::getSimpleLayerCost(mlir::Operation* operation,
                                                const VPUNNCostParameters& parameters) const {
    auto outputType = operation->getResult(0).getType().cast<vpux::NDTypeInterface>();

    return outputType.getTotalAllocSize().count() / getNumClusterCorrectionSize(parameters._strategy);
}

StrategyCost LayerVPUNNCost::getNCELayerCost(VPU::NCEOpInterface nceOp, const VPUNNCostParameters& parameters) const {
    const auto costParam = VPU::getWorkloadCostParam(nceOp, _arch, _numDPUs);
    SmallVector<VPUNN::DPULayer> vpunnLayers{VPU::getDPULayer(costParam)};

    // Check CMX memory as VPUNN works with layer which fits CMX memory
    auto isOne = [&](auto i) {
        return i == 1;
    };

    if (!parameters._tiling.empty() && !llvm::all_of(parameters._tiling.front().axis, isOne)) {
        _log.trace("Tiling op {0} to fit into cmx before passing to VPUNN Layer API", nceOp.getLoc());
        auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
        VPUX_THROW_WHEN(tilingOp == nullptr, "NCE op {0} at {1} should be a tiling op", nceOp->getName(),
                        nceOp.getLoc());

        auto tilingVPUNNLayer = [&](const VPUNN::DPULayer& vpunnLayer,
                                    const OutputTiling& outTiles) -> SmallVector<VPUNN::DPULayer> {
            SmallVector<VPUNN::DPULayer> vpunnLayers;
            for (auto& outTile : outTiles) {
                vpunnLayers.push_back(vpunnLayer);
                auto inTiles = tilingOp.backInferTileInfo(outTile, _log);
                auto& inputTile = inTiles.tiles.front();
                auto inPad = inTiles.pads;
                vpunnLayers.back().inputs = {getVPUTensor(inputTile.shape, costParam.inDataType)};
                vpunnLayers.back().outputs = {getVPUTensor(outTile.shape, costParam.outDataType)};
                if (inPad.hasValue()) {
                    vpunnLayers.back().padding = {
                            static_cast<unsigned int>(inPad->top), static_cast<unsigned int>(inPad->bottom),
                            static_cast<unsigned int>(inPad->left), static_cast<unsigned int>(inPad->right)};
                }
            }
            return vpunnLayers;
        };
        vpunnLayers = tilingVPUNNLayer(vpunnLayers[0], parameters._tiling);
    }

    auto vpunnStrategy =
            VPU::getVPULayerStrategy(parameters._strategy, _numDPUs, _numClusters, _numShaveActs, parameters._prefetch);

    _log.trace("Start calculating vpunn cost for Op {0} with strategy {1}", nceOp.getLoc(), parameters._strategy);
    SmallVector<StrategyCost> vpunnLayerCosts;
    vpunnLayerCosts.reserve(vpunnLayers.size());
    for (auto& vpunnLayer : vpunnLayers) {
        StrategyCost cost = checkAndReturnCost(_vpunnCostModel->Layer(vpunnLayer, vpunnStrategy), _log);
        if (cost >= VPU::INVALID_COST_BASE) {
            printVPUNNLayerConfig(vpunnLayer, vpunnStrategy, _log);

            if (cost == VPU::ERROR_INPUT_TOO_BIG && !vpunnLayerCosts.empty()) {
                _log.trace(" Use the first availabe layer cost to estimate the layer with ERROR_INPUT_TOO_BIG");
                cost = vpunnLayerCosts.front();
            } else {
                return std::numeric_limits<VPU::StrategyCost>::max();
            }
        }
        _log.trace(" VPUNN layer cost {0}", cost);
        vpunnLayerCosts.push_back(cost);
    }

    const auto vpunnCost = std::accumulate(vpunnLayerCosts.begin(), vpunnLayerCosts.end(), 0);

    _log.trace("VPUNN total layer cost {0}", vpunnCost);
    return vpunnCost;
}

StrategyCost LayerVPUNNCost::getSWLayerCost(VPU::SWOpInterface swOp, const VPUNNCostParameters& parameters) const {
    const auto vpunnLayer = getVPUNNSWKernelOp(swOp);
    if (!vpunnLayer) {
        return getSimpleLayerCost(swOp, parameters);
    }

    auto vpunnStrategy = VPU::getVPULayerStrategy(parameters._strategy, _numDPUs, _numClusters, _numShaveActs, false);
    return _vpunnCostModel->Layer(*vpunnLayer, vpunnStrategy);
}

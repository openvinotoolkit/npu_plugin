//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/operation_strategies.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

namespace vpux::VPU {

struct VPUNNCostParameters {
    VPUNNCostParameters(VPU::MultiClusterStrategy strategy, const OutputTiling& tiling = {}, bool prefetch = false)
            : _strategy(strategy), _tiling(tiling), _prefetch(prefetch) {
    }

    VPU::MultiClusterStrategy _strategy;
    OutputTiling _tiling;
    bool _prefetch;
};

/*
 *  Class adaptor to get cost from VPUNN
 *  for DPU, SW layers
 */

class LayerVPUNNCost final {
public:
    LayerVPUNNCost(mlir::func::FuncOp func, Logger log = Logger::global()): _log(log) {
        auto module = func->getParentOfType<mlir::ModuleOp>();
        _arch = VPU::getArch(module);
        _vpunnCostModel = VPU::createLayerCostModel(_arch);

        auto nceEngine = IE::getAvailableExecutor(module, ExecutorKind::NCE);
        auto dpuExec = nceEngine.getSubExecutor(VPU::ExecutorKind::DPU);
        _numClusters = nceEngine.count();
        _numDPUs = dpuExec.count();
        _vpuDevice = getVPUDeviceType(_arch);
        if (auto shaveActExec = nceEngine.getSubExecutor(ExecutorKind::SHAVE_ACT)) {
            _numShaveActs = shaveActExec.count();
        }
    };

    /*
     *  Get the cost for operation for particular parameters
     */
    StrategyCost getStrategyCost(mlir::Operation* operation, const VPUNNCostParameters& parameters) const;

private:
    /*
     *  Get the cost of NCE operation.
     *   In case tiling is passed, cost is taken with tiling parameters
     */
    StrategyCost getNCELayerCost(VPU::NCEOpInterface nceOp, const VPUNNCostParameters& parameters) const;

    /*
     *  Get cost of SW kernels
     */
    StrategyCost getSWLayerCost(VPU::SWOpInterface swOp, const VPUNNCostParameters& parameters) const;

    /*
     *  Get simple cycle cost for operation which is not supported by VPUNN yet
     *  Approximate cost is size of output tensor in bytes per cluster
     */
    StrategyCost getSimpleLayerCost(mlir::Operation* operation, const VPUNNCostParameters& parameters) const;

    /*
     *  Get divisor to get size for output tensor per cluster
     */
    size_t getNumClusterCorrectionSize(VPU::MultiClusterStrategy strategy) const;

    VPU::ArchKind _arch;
    int64_t _numClusters;
    int64_t _numDPUs;
    int64_t _numShaveActs;
    VPUNN::VPUDevice _vpuDevice;
    std::shared_ptr<VPUNN::VPULayerCostModel> _vpunnCostModel;
    Logger _log;
};

}  // namespace vpux::VPU

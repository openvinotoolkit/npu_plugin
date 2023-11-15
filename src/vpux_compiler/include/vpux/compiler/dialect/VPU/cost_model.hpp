//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"

#include <vpu_cost_model.h>
#include <vpu_layer_cost_model.h>

#include <memory>

namespace vpux {

float getWeightsSparsityRatio(vpux::NDTypeInterface weightsType, int64_t compressedSize);

namespace VPU {
static constexpr uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
// The top 100 maximum UINT32 vals for error codes
static constexpr uint32_t INVALID_COST_BASE = MAX_VAL - 100;
static constexpr uint32_t ERROR_INPUT_TOO_BIG = MAX_VAL - 0;

std::shared_ptr<VPUNN::VPUCostModel> createCostModel(ArchKind arch);
std::shared_ptr<VPUNN::VPULayerCostModel> createLayerCostModel(ArchKind arch, bool isFastModel = true);
uint32_t checkAndReturnCost(const VPUNN::CyclesInterfaceType& cost, vpux::Logger log, bool beSilent = false);
void printVPUNNLayerConfig(const VPUNN::DPULayer& layer, const VPUNN::VPULayerStrategy& strategy, vpux::Logger log);
float getWeightsSparsityRatio(mlir::Value weights);
VPUNN::VPUDevice getVPUDeviceType(VPU::ArchKind archKind);
VPUNN::DataType getVPUNNElementType(mlir::Type type);
VPUNN::VPUTensor getVPUTensor(ShapeRef shape, mlir::Type elemType);
VPUNN::ExecutionMode getExecutionMode(VPU::MPEMode mpeMode);
VPUNN::ActivationFunction getVPUNNActivationFunction(VPU::PPETaskAttr ppeTask);
VPUNN::VPULayerStrategy getVPULayerStrategy(VPU::MultiClusterStrategy, size_t nDPUs, size_t nTiles, size_t nSHVs = 1,
                                            bool prefetching = false);
VPUNN::DPULayer getDPULayer(const VPUIP::WorkloadCostParams& params);
VPUNN::DPUWorkload getDPUWorkload(const VPUIP::WorkloadCostParams& tileParams, const VPUIP::WorkloadTile& wl);
VPUIP::WorkloadCostParams getWorkloadCostParam(VPU::NCEOpInterface nceOp, VPU::ArchKind arch, int64_t numDPU,
                                               int64_t numTiles = 1);

}  // namespace VPU
}  // namespace vpux

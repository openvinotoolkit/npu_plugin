//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"

#include <vpu_cost_model.h>
#include <vpu_layer_cost_model.h>

#include <memory>

namespace vpux {
namespace VPU {
uint32_t static constexpr INVALID_COST = std::numeric_limits<uint32_t>::max();

std::shared_ptr<VPUNN::VPUCostModel> createCostModel(ArchKind arch);
std::shared_ptr<VPUNN::VPULayerCostModel> createLayerCostModel(ArchKind arch, bool isFastModel = true);
uint32_t checkAndReturnCost(const VPUNN::CyclesInterfaceType& cost, vpux::Logger log, bool beSilent = false);
void printVPUNNLayerConfig(const VPUNN::DPULayer& layer, const VPUNN::VPULayerStrategy& strategy);
float getWeightsSparsityRatio(mlir::Value weights);
VPUNN::VPUDevice getVPUDeviceType(VPU::ArchKind archKind);
VPUNN::DataType getVPUNNElementType(mlir::Type type);
VPUNN::VPUTensor getVPUTensor(ShapeRef shape, mlir::Type elemType);
VPUNN::VPULayerStrategy getVPULayerStrategy(VPU::MultiClusterStrategy, size_t nDPUs, size_t nTiles, size_t nSHVs = 1,
                                            bool prefetching = false);
VPUNN::DPULayer getDPULayer(const VPUIP::WorkloadCostParams& params);
VPUIP::WorkloadCostParams getWorkloadCostParam(VPU::NCEOpInterface nceOp, VPU::ArchKind arch, int64_t numDPU);

}  // namespace VPU
}  // namespace vpux

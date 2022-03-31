//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/cost_model_data.hpp"

using namespace vpux;

namespace {

ArrayRef<char> getCostModelData(VPU::ArchKind archKind) {
    switch (archKind) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX311X:
        return makeArrayRef(VPU::COST_MODEL_2_0, VPU::COST_MODEL_2_0_SIZE);
    case VPU::ArchKind::VPUX37XX:
        return makeArrayRef(VPU::COST_MODEL_2_7, VPU::COST_MODEL_2_7_SIZE);
    default:
        VPUX_THROW("Unsupported VPU arch type: '{0}'", archKind);
    }
}

}  // namespace

std::shared_ptr<VPUNN::VPUCostModel> vpux::VPU::createCostModel(ArchKind arch) {
    const auto costModelData = getCostModelData(arch);
    return std::make_shared<VPUNN::VPUCostModel>(costModelData.data(), costModelData.size(), false);
}

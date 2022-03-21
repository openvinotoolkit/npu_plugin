//
// Copyright 2022 Intel Corporation.
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

#include "vpux/compiler/dialect/VPU/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/cost_model_data.hpp"

using namespace vpux;

namespace {

ArrayRef<char> getCostModelData(VPU::ArchKind archKind) {
    switch (archKind) {
    case VPU::ArchKind::KMB:
    case VPU::ArchKind::TBH:
        return makeArrayRef(VPU::COST_MODEL_2_0, VPU::COST_MODEL_2_0_SIZE);
    case VPU::ArchKind::MTL:
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

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include <npu_37xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU37XX/types.hpp"

using namespace npu37xx;

struct Vpu37XXBarrierCfg {
    nn_public::VpuBarrierCountConfig barrier;
};

#define CREATE_HW_BARRIER_DESC(field, value)                                       \
    [] {                                                                           \
        Vpu37XXBarrierCfg hwBarrierDesc;                                           \
        memset(reinterpret_cast<void*>(&hwBarrierDesc), 0, sizeof(hwBarrierDesc)); \
        hwBarrierDesc.field = value;                                               \
        return hwBarrierDesc;                                                      \
    }()

class VPU37XX_VpuBarrierCountConfigTest :
        public MLIR_RegMappedVPU37XXUnitBase<Vpu37XXBarrierCfg, vpux::VPU37XX::RegMapped_VpuBarrierCountConfigType> {};

TEST_P(VPU37XX_VpuBarrierCountConfigTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, Vpu37XXBarrierCfg>> barrierFieldSetVPU37XX = {
        {{
                 {"next_same_id_", {{"next_same_id_", 0xFFFFFFFF}}},
         },
         CREATE_HW_BARRIER_DESC(barrier.next_same_id_, 0xFFFFFFFF)},
        {{
                 {"producer_count_", {{"producer_count_", 0xFFFF}}},
         },
         CREATE_HW_BARRIER_DESC(barrier.producer_count_, 0xFFFF)},
        {{
                 {"consumer_count_", {{"consumer_count_", 0xFFFF}}},
         },
         CREATE_HW_BARRIER_DESC(barrier.consumer_count_, 0xFFFF)},
        {{
                 {"real_id_", {{"real_id_", 0xFF}}},
         },
         CREATE_HW_BARRIER_DESC(barrier.real_id_, 0xFF)},
};

INSTANTIATE_TEST_CASE_P(VPU37XX_MappedRegs, VPU37XX_VpuBarrierCountConfigTest,
                        testing::ValuesIn(barrierFieldSetVPU37XX));

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include <npu_37xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU37XX/types.hpp"

using namespace npu37xx;

struct Vpu37ActKernelRange {
    nn_public::VpuActKernelRange actKernelRange;
};

#define CREATE_HW_ACT_KERNEL_RANGE_DESC(field, value)                                            \
    [] {                                                                                         \
        Vpu37ActKernelRange hwActKernelRangeDesc;                                                \
        memset(reinterpret_cast<void*>(&hwActKernelRangeDesc), 0, sizeof(hwActKernelRangeDesc)); \
        hwActKernelRangeDesc.field = value;                                                      \
        return hwActKernelRangeDesc;                                                             \
    }()

class VPU37XX_VpuActKernelRangeTest :
        public MLIR_RegMappedVPU37XXUnitBase<Vpu37ActKernelRange, vpux::VPU37XX::RegMapped_VpuActKernelRangeType> {};

TEST_P(VPU37XX_VpuActKernelRangeTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, Vpu37ActKernelRange>> actKernelRangeFieldSetVPU37XX = {
        {{
                 {"type", {{"type", 0x07}}},
         },
         CREATE_HW_ACT_KERNEL_RANGE_DESC(actKernelRange.type, nn_public::VpuActWLType::WL_CACHE_OP_FLUSHINV)},
        {{
                 {"kernel_entry", {{"kernel_entry", 0xFFFFFFFF}}},
         },
         CREATE_HW_ACT_KERNEL_RANGE_DESC(actKernelRange.kernel_entry, 0xFFFFFFFF)},
        {{
                 {"text_window_base", {{"text_window_base", 0xFFFFFFFF}}},
         },
         CREATE_HW_ACT_KERNEL_RANGE_DESC(actKernelRange.text_window_base, 0xFFFFFFFF)},
        {{
                 {"code_size", {{"code_size", 0xFFFFFFFF}}},
         },
         CREATE_HW_ACT_KERNEL_RANGE_DESC(actKernelRange.code_size, 0xFFFFFFFF)},
        {{
                 {"data_sec_size", {{"data_sec_size", 0xFFFFFFFF}}},
         },
         CREATE_HW_ACT_KERNEL_RANGE_DESC(actKernelRange.data_sec_size, 0xFFFFFFFF)},
        {{
                 {"kernel_invo_count", {{"kernel_invo_count", 0xFFFFFFFF}}},
         },
         CREATE_HW_ACT_KERNEL_RANGE_DESC(actKernelRange.kernel_invo_count, 0xFFFFFFFF)},
};

INSTANTIATE_TEST_CASE_P(VPU37XX_MappedRegs, VPU37XX_VpuActKernelRangeTest,
                        testing::ValuesIn(actKernelRangeFieldSetVPU37XX));

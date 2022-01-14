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

#include <nn_memory_map.h>

namespace nn {
namespace common_runtime {
struct NNCmxMemoryMap : public util::MemoryMap {
    struct Slice {
        struct {
            Fragment<15_KB> snn_data_;
            Fragment<1_KB> snn_stack_;
        } global_;
        struct {
            Fragment<64_KB> metadata_;
            Fragment<1928_KB> workspace_;
            Fragment<8_KB> actshv_stack_;
            Fragment<32_KB> dma_storage_;
        } user_;
    } slice_[MAX_TILES];

    static_assert(sizeof(Slice) == 2_MB, "Invalid layout for slices");

    // Stolen window assertions
    static_assert(sizeof(slice_[0].global_) >= common_runtime::STOLEN_WINDOW_MIN_LEN,
                  "Stolen window must be at least 512B");
    static_assert(sizeof(slice_[0].global_) <= common_runtime::SLICE_LENGTH / 2, "Stolen window maximum size exceeded");
    static_assert(!(sizeof(slice_[0].global_) & (sizeof(slice_[0].global_) - 1)),
                  "Stolen window size must be a power of two");
    static_assert(!(offsetof(NNCmxMemoryMap::Slice, global_) % sizeof(slice_[0].global_)),
                  "Stolen window must be window size aligned");
};
} // namespace common_runtime
} // namespace nn

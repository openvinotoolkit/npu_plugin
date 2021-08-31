/*
* {% copyright %}
*/
#ifndef NN_CMX_MEMORY_MAP_H_
#define NN_CMX_MEMORY_MAP_H_

#include <nn_memory_map.h>

namespace nn
{
    namespace inference_runtime
    {
        struct NNCmxMemoryMap : public util::MemoryMap
        {
#if !defined(CONFIG_TARGET_SOC_3600) && !defined(CONFIG_TARGET_SOC_3710) && !defined(CONFIG_TARGET_SOC_3720)
            struct Slice0
            {
                Fragment<15_KB> snn_data_;
                Fragment<1_KB> reserved_;
                Fragment<48_KB> metadata_;
                Fragment<896_KB> workspace_;
                Fragment<64_KB> lnn_data_;
            } slice0_;

            struct Slice1
            {
                Fragment<64_KB> lnn_data_;
                Fragment<896_KB> workspace_;
                Fragment<48_KB> metadata_;
                Fragment<1_KB> reserved_;
                Fragment<15_KB> snn_data_;
            } slice1_;

            struct Slice2
            {
                Fragment<15_KB> snn_data_;
                Fragment<1_KB> reserved_;
                Fragment<48_KB> metadata_;
                Fragment<896_KB> workspace_;
                Fragment<64_KB> lnn_data_;
            } slice2_;

            struct Slice3
            {
                Fragment<64_KB> lnn_data_;
                Fragment<896_KB> workspace_;
                Fragment<48_KB> metadata_;
                Fragment<1_KB> reserved_;
                Fragment<15_KB> snn_data_;
            } slice3_;

            static_assert(sizeof(slice0_) == 1_MB, "Invalid layout for slice 0");
            static_assert(sizeof(slice1_) == 1_MB, "Invalid layout for slice 1");
            static_assert(sizeof(slice2_) == 1_MB, "Invalid layout for slice 2");
            static_assert(sizeof(slice3_) == 1_MB, "Invalid layout for slice 3");

            static_assert(sizeof(slice0_.workspace_) == sizeof(slice1_.workspace_), "Workspace size mismatch 0-1");
            static_assert(sizeof(slice1_.workspace_) == sizeof(slice2_.workspace_), "Workspace size mismatch 1-2");
            static_assert(sizeof(slice2_.workspace_) == sizeof(slice3_.workspace_), "Workspace size mismatch 2-3");

            static_assert(offsetof(Slice0, workspace_) == offsetof(Slice1, workspace_), "Workspace offset mismatch 0-1");
            static_assert(offsetof(Slice1, workspace_) == offsetof(Slice2, workspace_), "Workspace offset mismatch 1-2");
            static_assert(offsetof(Slice2, workspace_) == offsetof(Slice3, workspace_), "Workspace offset mismatch 2-3");
#else
            struct Slice0
            {
                Fragment<15_KB> snn_data_;
                Fragment<1_KB> reserved_;
                Fragment<64_KB> metadata_;
                Fragment<1936_KB> workspace_;
                Fragment<32_KB> lnn_data_;
            } slice0_;

            struct Slice1
            {
                Fragment<32_KB> lnn_data_;
                Fragment<1936_KB> workspace_;
                Fragment<64_KB> metadata_;
                Fragment<1_KB> reserved_;
                Fragment<15_KB> snn_data_;
            } slice1_;

            static_assert(sizeof(slice0_) == 2_MB, "Invalid layout for slice 0");
            static_assert(sizeof(slice1_) == 2_MB, "Invalid layout for slice 1");

            static_assert(sizeof(slice0_.workspace_) == sizeof(slice1_.workspace_), "Workspace size mismatch 0-1");
#endif
        };
    }
}

#endif // NN_CMX_MEMORY_MAP_H_

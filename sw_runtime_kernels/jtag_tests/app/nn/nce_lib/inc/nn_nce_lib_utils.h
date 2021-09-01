/*
* {% copyright %}
*/
#ifndef NN_NCE_LIB_UTILS_H
#define NN_NCE_LIB_UTILS_H

#include <nn_runtime_types.h>
#include <mv_types.h>

#include <graphfile_generated.h>

namespace nn
{
    namespace nce_lib
    {
        void SetupInvariant_Input_SE_size(const MVCNN::NCEInvariantFields *wl_invariant, dpu_runtime::DPUInvariantRegisters &invariantRegisterst);
        unsigned int SOH_LinesPerCluster(unsigned int parentHeight, unsigned int height, unsigned int clusters);

        void SetupInvariant_SOH(const MVCNN::NCEInvariantFields *wl_invariant,
                 nn::dpu_runtime::DPUInvariantRegisters &invariantRegisters,
                 uint32_t clusters);
        void SetupInvariant_SOH_Input(const MVCNN::NCEInvariantFields *wl_invariant,
                 nn::dpu_runtime::DPUInvariantRegisters &invariantRegisters);
        unsigned int SetupVariant_SOH(const MVCNN::NCEInvariantFields *fb_invariant, const MVCNN::NCEVariantFields *fb_variant,
                 dpu_runtime::DPUInvariant& invariant, dpu_runtime::DPUVariant &variant, unsigned int clusters);
        void Setup_Output_SOH(const MVCNN::NCEInvariantFields *fb_invariant, dpu_runtime::DPUInvariantRegisters &invariant, bool is_out_dense);
        void Update_Invariant_SOH(dpu_runtime::DPULayerTypes opType, dpu_runtime::DPUInvariant &invariant, inference_runtime::RelativeAddress &input, const inference_runtime::NNRelocationData &relocationData);
        bool Is_Dtype_Mix_Supported(MVCNN::DType inputType, MVCNN::DType weightsType);

        void SetupInvariant_Input(dpu_runtime::DPUInvariantRegisters &registers, const MVCNN::TensorReference *tensor);
        void SetupInvariant_Output(dpu_runtime::DPUInvariantRegisters &registers, const MVCNN::TensorReference *tensor);

#if (defined(CONFIG_TARGET_SOC_3600) || defined(CONFIG_TARGET_SOC_3710) || defined(CONFIG_TARGET_SOC_3720))
        void SetupVariant_NTHW_NTK(const MVCNN::NCEInvariantFields *fb_invariant, dpu_runtime::DPUVariantRegisters &variant);
        void SetupInvariant_Grid(const MVCNN::NCEInvariantFields *fb_invariant, dpu_runtime::DPUInvariantRegisters &invariant);
#endif // CONFIG_TARGET_SOC_3600 || CONFIG_TARGET_SOC_3710 || CONFIG_TARGET_SOC_3720
    }
}
#endif // NN_NCE_LIB_UTILS_H

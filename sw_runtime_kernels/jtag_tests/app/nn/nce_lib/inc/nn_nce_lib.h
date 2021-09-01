/*
* {% copyright %}
*/
#ifndef NN_NCE_LIB_H_
#define NN_NCE_LIB_H_

#include <nn_runtime_types.h>
#include <graphfile_generated.h>

namespace nn
{
    namespace nce_lib
    {
        class DPUConfig
        {
        public:
            DPUConfig(const MVCNN::NCEInvariantFields *invariant, const MVCNN::NCEVariantFields *variant, unsigned int cluster_count);

            bool Setup_Invariant(dpu_runtime::DPULayerTypes &opType, dpu_runtime::DPUInvariant& invariant, dpu_runtime::DPUAddresses &addresses);
            bool Setup_Variant(dpu_runtime::DPUVariant& variant, dpu_runtime::DPUInvariant& invariant);

            bool workaround_for_bug_33243() const;

        private:
            const MVCNN::NCEInvariantFields *fb_invariant_;
            const MVCNN::NCEVariantFields *fb_variant_;
            const dpu_runtime::DPULayerTypes opType_;
            const unsigned int cluster_count_;

            unsigned int ConfigWorkloadSize(unsigned int size) const;
            unsigned int ConfigWorkloadStart(unsigned int start) const;

            bool Setup_Input(dpu_runtime::DPUInvariantRegisters& invariant);
            bool Setup_Weights(dpu_runtime::DPUInvariantRegisters& invariant);
            bool Setup_Kernel(dpu_runtime::DPUInvariantRegisters& invariant);
            bool Setup_Output(dpu_runtime::DPUInvariantRegisters& invariant);
            bool SetupInvariant_RelativeAddresses(dpu_runtime::DPUAddresses& addresses);

            bool SetupInvariant_CMConv(dpu_runtime::DPUInvariantRegisters& registers);
            void SetupInvariant_Convolution(dpu_runtime::DPUInvariantRegisters& registers);
            void SetupInvariant_DwConvolution(dpu_runtime::DPUInvariantRegisters& registers);
            bool SetupInvariant_Eltwise(dpu_runtime::DPUInvariantRegisters& registers);
            void SetupInvariant_MaxPool(dpu_runtime::DPUInvariantRegisters& registers);

            bool Setup_PPE(dpu_runtime::DPUInvariant& invariant);

            bool apply_workaround_for_bug_33243(dpu_runtime::DPUInvariantRegisters& registers);
            void apply_16x1_grid_limit(dpu_runtime::DPUInvariantRegisters& registers, const char *layer);
        };

        bool Update_Invariant(dpu_runtime::DPULayerTypes opType, dpu_runtime::DPUInvariant &invariant,
                              const dpu_runtime::DPUAddresses &addresses, const inference_runtime::NNRelocationData &relocationData);
    }
}

#endif // NN_NCE_LIB_H_

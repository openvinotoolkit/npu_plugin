/*
* {% copyright %}
*/
#include "nn_nce_lib.h"
#include "nn_nce_lib_conversion_fbs.h"
#include "nn_runtime_types.h"

#include <math.h>

namespace nn
{
    namespace nce_lib
    {
        bool DPUConfig::SetupInvariant_CMConv(dpu_runtime::DPUInvariantRegisters& registers)
        {
            registers.tensor_size0.tensor_size0_bf.tensor_size_y = fb_invariant_->input_data()->dimensions()->Get(Y);
            registers.tensor_size1.tensor_size1_bf.tensor_size_z = 16;

            // FIXME: handle float strides
            auto y_stride = fb_invariant_->input_data()->strides()->Get(STRIDES(Y));
            float y_stridef = (float)y_stride;

            if(floorf(y_stridef) != y_stridef)
                nnLog(MVLOG_WARN, "Truncating y-stride %f. Sub-byte strides are unsupported.", y_stridef);

            if ((uint32_t)(y_stride) % 16)
                nnLog(MVLOG_WARN, "CM Conv requires Y stride to be multiple to 16. Received: %u",
                    (uint32_t)(y_stride));

            // CM Conv requires act_dense == 0, even if activations are dense.
            // registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense = 0;
            auto wt_tensor_ref = fb_invariant_->weights_data();
            bool is_wt_dense = wt_tensor_ref->data()->sparsity_index() == DEFAULT_INDEX;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense = is_wt_dense;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en = 1;

            // Dedicated HW for channel-major removed for 2.7 - all convolutions
            // processed as Z-major (or subset of this functionality)
            registers.tensor_mode.tensor_mode_bf.dw_input = 0;
            registers.tensor_mode.tensor_mode_bf.cm_input = 0;
            registers.tensor_mode.tensor_mode_bf.zm_input = 1;

            registers.kernel_pad_cfg.kernel_pad_cfg_bf.layer1_wt_sp_ins = 1; // enable the IDU to generate the weight sparsity
            registers.z_config.z_config_bf.cm_sp_pattern = 7; // num channel bit enable

            // Need to tell hw we're compressing input formated data
            if (fb_invariant_->parent_input_tensor()->dimensions()->Get(Z) == 4 )
                registers.kernel_pad_cfg.kernel_pad_cfg_bf.layer1_cmp_en = 1;

            if (fb_invariant_->output_data()->data_dtype() == MVCNN::DType::DType_FP16) {
                registers.odu_cfg.odu_cfg_bf.grid = dpu_runtime::ODU_GRID_4x4;
            }

            return true;
        }
    }
}

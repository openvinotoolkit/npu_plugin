/*
* {% copyright %}
*/
#include "nn_nce_lib.h"
#include "nn_nce_lib_conversion_fbs.h"
#include "nn_nce_lib_utils.h"

namespace nn
{
    namespace nce_lib
    {
        bool DPUConfig::SetupInvariant_Eltwise(dpu_runtime::DPUInvariantRegisters& registers)
        {
            if (fb_invariant_->kernelW() != 1 ||
                fb_invariant_->kernelH() != 1)
            {
                nnLog(MVLOG_ERROR, "Eltwise only supports 1x1 kernel. Got %ux%u", fb_invariant_->kernelW(), fb_invariant_->kernelH());
                return false;
            }

            if (fb_invariant_->parent_output_tensor() &&
                fb_invariant_->parent_output_tensor()->dimensions()->Get(Z) != fb_invariant_->output_data()->dimensions()->Get(Z))
            {
                nnLog(MVLOG_ERROR, "Eltwise does not support split over K\n");
                return false;
            }

            auto in_tensor_ref  = fb_invariant_->input_data();
            auto out_tensor_ref = fb_invariant_->output_data();
            auto wt_tensor_ref  = fb_invariant_->weights_data();

            auto amode = fb_invariant_->input_data()->data_dtype();
            auto wmode = fb_invariant_->weights_data()->data_dtype();
            auto omode = fb_invariant_->output_data()->data_dtype();

            registers.tensor_mode.tensor_mode_bf.dw_input = 0;
            registers.tensor_mode.tensor_mode_bf.cm_input = 0;
            registers.tensor_mode.tensor_mode_bf.zm_input = 1;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en = 1;

            // Use the unsigned short input & weight scale factors for elop_scale_a/b
            // PPE will need to apply a shift right to compensate: see the ELTWISE specialization in ppe_task.cpp
            // NOTE: VPU2p7 HW feature allowing different quantization scales is not supported yet
            // So this branch is forced off with 0, even if the blob gives these params
            // See EISW-5671
            if (0 && in_tensor_ref->quant_mult()->size() && wt_tensor_ref->quant_mult()->size() )
            {
                unsigned short aMult  = in_tensor_ref->quant_mult()->Get(0);
                unsigned short bMult  = wt_tensor_ref->quant_mult()->Get(0);
                unsigned short aShift = in_tensor_ref->quant_shift()->Get(0);
                unsigned short bShift = wt_tensor_ref->quant_shift()->Get(0);
                unsigned short outMult = out_tensor_ref->quant_mult()->Get(0);
                unsigned short outShift = out_tensor_ref->quant_shift()->Get(0);

                uint64_t fScaleFactor1 = static_cast<uint64_t>(aMult*outMult)<<15;
                uint64_t fScaleFactor2 = static_cast<uint64_t>(bMult*outMult)<<15;
                unsigned scaleFactor1Shift = aShift + outShift;
                unsigned scaleFactor2Shift = bShift + outShift;

                unsigned short elopScaleA = static_cast<unsigned short>(fScaleFactor1 >> scaleFactor1Shift);
                unsigned short elopScaleB = static_cast<unsigned short>(fScaleFactor2 >> scaleFactor2Shift);

                registers.elop_scale.elop_scale_bf.elop_scale_a = elopScaleA;
                registers.elop_scale.elop_scale_bf.elop_scale_b = elopScaleB;
            }
            else
            {
                registers.elop_scale.elop_scale_bf.elop_scale_a = 1;
                registers.elop_scale.elop_scale_bf.elop_scale_b = 1;
            }

            // For FP16 eltwise grid needs to be 4x4
            if(((amode == MVCNN::DType::DType_FP16) && (wmode == MVCNN::DType::DType_FP16)) || (omode == MVCNN::DType::DType_FP16))
            {
                registers.odu_cfg.odu_cfg_bf.grid = dpu_runtime::ODU_GRID_4x4;
                registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = dpu_runtime::MPE_GRID_4x4;
            }

            bool is_wt_dense = wt_tensor_ref->data()->sparsity_index() == DEFAULT_INDEX;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense = is_wt_dense;

            registers.elops_wload.elops_wload_bf.elop_wload = 1;//read in 2 tensors instead of a tensor and weight sets for a standard convolution.

            SetupInvariant_SOH_Input(fb_invariant_, registers);
            SetupInvariant_Input_SE_size(fb_invariant_, registers);

            return true;
        }
    }
}

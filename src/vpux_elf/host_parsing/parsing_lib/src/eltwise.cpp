/*
 * {% copyright %}
 */
#include <convert.h>
#include <nn_logging.h>

namespace parsing_lib {
    using namespace host_parsing;

bool DPUConfigurator::SetupInvariant_Eltwise(DPUInvariantRegisters &registers) {
    if (srcInvariant.kernelW != 1 || srcInvariant.kernelH != 1) {
        nnLog(MVLOG_ERROR, "Eltwise only supports 1x1 kernel. Got %ux%u", srcInvariant.kernelW,
              srcInvariant.kernelH);
        return false;
    }

    if (srcInvariant.parent_output_tensor.valid() && srcInvariant.parent_output_tensor->dimensions[Z] !=
                                                     srcInvariant.output_data->dimensions[Z]) {
        nnLog(MVLOG_ERROR, "Eltwise does not support split over K\n");
        return false;
    }

    auto in_tensor_ref = srcInvariant.input_data;
    auto out_tensor_ref = srcInvariant.output_data;
    auto wt_tensor_ref = srcInvariant.weights_data;

    auto amode = srcInvariant.input_data->data_dtype;
    auto wmode = srcInvariant.weights_data->data_dtype;
    auto omode = srcInvariant.output_data->data_dtype;

    registers.tensor_mode.tensor_mode_bf.dw_input = 0;
    registers.tensor_mode.tensor_mode_bf.cm_input = 0;
    registers.tensor_mode.tensor_mode_bf.zm_input = 1;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en = 1;

    // Use the unsigned short input & weight scale factors for elop_scale_a/b
    // PPE will need to apply a shift right to compensate: see the ELTWISE specialization in ppe_task.cpp
    // NOTE: VPU2p7 HW feature allowing different quantization scales is not supported yet
    // So this branch is forced off with 0, even if the blob gives these params
    // See EISW-5671
    if (0 && in_tensor_ref->quant_mult.size() && wt_tensor_ref->quant_mult.size()) {
        unsigned short aMult = in_tensor_ref->quant_mult[0];
        unsigned short bMult = wt_tensor_ref->quant_mult[0];
        unsigned short aShift = in_tensor_ref->quant_shift[0];
        unsigned short bShift = wt_tensor_ref->quant_shift[0];
        unsigned short outMult = out_tensor_ref->quant_mult[0];
        unsigned short outShift = out_tensor_ref->quant_shift[0];

        uint64_t fScaleFactor1 = static_cast<uint64_t>(aMult * outMult) << 15;
        uint64_t fScaleFactor2 = static_cast<uint64_t>(bMult * outMult) << 15;
        unsigned scaleFactor1Shift = aShift + outShift;
        unsigned scaleFactor2Shift = bShift + outShift;

        unsigned short elopScaleA = static_cast<unsigned short>(fScaleFactor1 >> scaleFactor1Shift);
        unsigned short elopScaleB = static_cast<unsigned short>(fScaleFactor2 >> scaleFactor2Shift);

        registers.elop_scale.elop_scale_bf.elop_scale_a = elopScaleA;
        registers.elop_scale.elop_scale_bf.elop_scale_b = elopScaleB;
    } else {
        registers.elop_scale.elop_scale_bf.elop_scale_a = 1;
        registers.elop_scale.elop_scale_bf.elop_scale_b = 1;
    }

    // For FP16 eltwise grid needs to be 4x4
    if (((amode == DType::FP16) && (wmode == DType::FP16)) ||
        (omode ==  DType::FP16)) {
        registers.odu_cfg.odu_cfg_bf.grid = to_underlying(ODUGrid::GRID_4x4);
        registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = to_underlying(MPEGrid::GRID_4x4);
    }

    bool is_wt_dense = wt_tensor_ref->data.sparsity_index == DEFAULT_INDEX;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense = is_wt_dense;

    // read in 2 tensors instead of a tensor and weight sets for a standard convolution.
    registers.elops_wload.elops_wload_bf.elop_wload = 1;

    SetupInvariant_SOH_Input(registers);
    SetupInvariant_Input_SE_size(registers);

    return true;
}
} // namespace nce_lib

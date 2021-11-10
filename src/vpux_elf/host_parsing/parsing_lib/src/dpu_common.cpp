/*
 * {% copyright %}
 */
#include <convert.h>
#include <nn_logging.h>
#include <dpu_common_utils.h>
#include <fp_utils.h>
#include <utils.h>

namespace {
// Round up val by N
template <size_t N>
uint32_t round_up(uint32_t t) {
    return static_cast<uint32_t>((t + N - 1) & ~(N - 1));
}
}

namespace parsing_lib {
    using namespace host_parsing;

constexpr unsigned int KERNEL_SIZE_MIN = 1;
constexpr unsigned int KERNEL_SIZE_MAX = 11;
constexpr unsigned int KERNEL_STRIDE_MIN = 1;
constexpr unsigned int KERNEL_STRIDE_MAX = 8;


#define CHECK_SETUP(f, p)                    \
    if (!f(p)) {                             \
        nnLog(MVLOG_ERROR, "%s failed", #f); \
        return false;                        \
    }

DPUConfigurator::DPUConfigurator(const Invariant &invariant, const Variant &variant,
                     unsigned int cluster_count)
    : srcInvariant(invariant)
    , srcVariant(variant)
    , cluster_count(cluster_count)
    , opType(srcInvariant.dpu_task_type) {}

bool DPUConfigurator::Setup_Invariant(DPUInvariant &invariant) {
    nnLog(MVLOG_DEBUG, "Layer type is %u", srcInvariant.dpu_task_type);

    invariant.output_sparsity_offset = 0;

    DPUInvariantRegisters &registers = invariant.registers;
    memset(&registers, 0, sizeof(registers));

    CHECK_SETUP(Setup_Input, registers)
    CHECK_SETUP(Setup_Weights, registers)
    CHECK_SETUP(Setup_Kernel, registers)
    CHECK_SETUP(Setup_Output, registers)

    switch (opType) {
        case DPULayerType::CONV:
            SetupInvariant_Convolution(registers);
            break;

        case DPULayerType::CMCONV:
            CHECK_SETUP(SetupInvariant_CMConv, registers)

            // Pre-compute the stride for CM conv to reduce SNN code size
            // Each line and plane has to be aligned to 16 bytes
            invariant.channel_major_stride =
                registers.tensor_size0.tensor_size0_bf.tensor_size_y *
                (round_up<16 * 8>(
                     registers.tensor_size0.tensor_size0_bf.tensor_size_x *
                     getIDUDTypeSizeBits(static_cast<InputTensorDType>(registers.tensor_mode.tensor_mode_bf.amode))) >>
                 3);
            break;

        case DPULayerType::MAXPOOL:
            SetupInvariant_MaxPool(registers);
            break;

        case DPULayerType::AVEPOOL:
            registers.elops_wload.elops_wload_bf.pool_wt_rd_dis=1;
        case DPULayerType::DWCONV:
            SetupInvariant_DwConvolution(registers);
            break;

        case DPULayerType::ELTWISE:
            CHECK_SETUP(SetupInvariant_Eltwise, registers)
            break;

        default:
            nnLog(MVLOG_WARN, "Unsupported dpu layer type: %d", static_cast<int>(opType));
            return false;
    }

    CHECK_SETUP(Setup_PPE, invariant);

    // TODO how does invariant get its cluster?
    //invariant.cluster_ = math::firstBitIndex(addresses.input_.index());

    return true;
}

// uint8_t DPUConfigurator::getCluster()
// {
//     return srcInvariant.input_data->locale_index[0];
// }

bool DPUConfigurator::Setup_Variant(DPUVariant &variant, DPUInvariant &invariant) {
    memset(reinterpret_cast<void *>(&variant.registers), 0, sizeof(DPUVariantRegisters));
    auto &in_tensor_ref = srcInvariant.input_data;

    auto stride_w = srcInvariant.kernel_strideW;
    auto stride_h = srcInvariant.kernel_strideH;
    auto K_w = srcInvariant.kernelW;
    auto K_h = srcInvariant.kernelH;

    auto global_PT = srcInvariant.kernel_padTop;
    auto global_PL = srcInvariant.kernel_padLeft;

    auto local_PT = srcVariant.padTop;
    auto local_PB = srcVariant.padBottom;
    auto local_PL = srcVariant.padLeft;
    auto local_PR = srcVariant.padRight;

    auto output_start_x = srcVariant.workload_start_X;
    auto output_start_y = srcVariant.workload_start_Y;
    auto output_start_z = srcVariant.workload_start_Z;

    auto output_end_x = srcVariant.workload_end_X;
    auto output_end_y = srcVariant.workload_end_Y;
    auto output_end_z = srcVariant.workload_end_Z;

    auto op_size_x = output_end_x - output_start_x + 1;
    auto op_size_y = output_end_y - output_start_y + 1;
    auto op_size_z = output_end_z - output_start_z + 1;

    variant.registers.weight_num = op_size_z;

    switch (opType) {
        case DPULayerType::CONV:
            variant.registers.weight_size =
                in_tensor_ref->dimensions[Z] * srcInvariant.kernelW * srcInvariant.kernelH;
            break;
        case DPULayerType::DWCONV:
        case DPULayerType::AVEPOOL:
        case DPULayerType::MAXPOOL:
            variant.registers.weight_size = op_size_z * srcInvariant.kernelW * srcInvariant.kernelH;
            break;
        case DPULayerType::ELTWISE:
            variant.registers.weight_size = in_tensor_ref->dimensions[X] * in_tensor_ref->dimensions[Y] *
                                             in_tensor_ref->dimensions[Z];
            break;
        case DPULayerType::CMCONV:
            variant.registers.weight_size = 16 * srcInvariant.kernelW * srcInvariant.kernelH;
            break;
        default:
            nnLog(MVLOG_FATAL, "Can't setup weight size. Layer type unknown : %u", static_cast<int>(opType));
    }

    nnLog(MVLOG_DEBUG,
          "OutStart X : %u OutStart Y : %u OutStart Z : %u OutEnd X : %u OutEnd "
          "Y : %u OutEnd Z : %u ",
          output_start_x, output_start_y, output_start_z, output_end_x, output_end_y, output_end_z);

    variant.registers.offset_addr.offset_addr_bf.dense_se =
        in_tensor_ref->data.sparsity_index != DEFAULT_INDEX;

    variant.registers.offset_addr.offset_addr_bf.conv_cond = srcInvariant.is_continued;

    variant.registers.workload_size0.workload_size0_bf.workload_size_x =
        stride_w * (op_size_x - 1) + K_w - local_PL - local_PR;
    variant.registers.workload_size0.workload_size0_bf.workload_size_y =
        stride_h * (op_size_y - 1) + K_h - local_PT - local_PB;
    variant.registers.workload_size1.workload_size1_bf.workload_size_z = ConfigWorkloadSize(op_size_z);
    variant.registers.workload_size1.workload_size1_bf.pad_count_up = local_PT;
    variant.registers.workload_size1.workload_size1_bf.pad_count_down = local_PB;
    variant.registers.workload_size1.workload_size1_bf.pad_count_left = local_PL;
    variant.registers.workload_size1.workload_size1_bf.pad_count_right = local_PR;
    variant.registers.workload_start0.workload_start0_bf.workload_start_x =
        (output_start_x * stride_w) - global_PL + local_PL;
    ;
    variant.registers.workload_start0.workload_start0_bf.workload_start_y =
        (output_start_y * stride_h) - global_PT + local_PT;
    variant.registers.workload_start1.workload_start1_bf.workload_start_z = ConfigWorkloadStart(output_start_z);

    variant.registers.te_beg1.te_beg1_bf.te_beg_x = output_start_x;
    variant.registers.te_beg0.te_beg0_bf.te_beg_y = output_start_y;
    variant.registers.te_beg0.te_beg0_bf.te_beg_z = output_start_z;

    variant.registers.te_end1.te_end1_bf.te_end_x = output_end_x;
    variant.registers.te_end0.te_end0_bf.te_end_y = output_end_y;
    variant.registers.te_end0.te_end0_bf.te_end_z = output_end_z;

    variant.weight_table_offset = output_start_z;

    if (opType == DPULayerType::DWCONV || opType == DPULayerType::MAXPOOL) {
        variant.registers.workload_start1.workload_start1_bf.workload_start_z = output_start_z;
        variant.registers.workload_size1.workload_size1_bf.workload_size_z = op_size_z;

    } else if (opType == DPULayerType::CMCONV || srcInvariant.parent_input_tensor->dimensions[Z] < 16) {
        variant.registers.workload_start1.workload_start1_bf.workload_start_z = 0;
        variant.registers.workload_size1.workload_size1_bf.workload_size_z = 16;
    } else {
        // All input channels required for one output channel
        variant.registers.workload_start1.workload_start1_bf.workload_start_z = 0;
        variant.registers.workload_size1.workload_size1_bf.workload_size_z =
            srcInvariant.input_data->dimensions[Z];
    }

    // Split over K, and also streaming over K for now ....
    // ODU has a view of the full output tensor, yet as an optimization
    // in each cluster we bring weights and weight_table portions for each
    // output channel subset we compute in that particular cluster
    if (srcInvariant.output_data->dimensions[Z] !=
        srcInvariant.parent_output_tensor->dimensions[Z]) {
        if (srcInvariant.output_data->locale_index.size() > 1) {
            nnLog(MVLOG_INFO, "Using symmetric SoK: %u instead of %u",
                  variant.weight_table_offset % srcInvariant.output_data->dimensions[Z],
                  variant.weight_table_offset);

            variant.weight_table_offset %= srcInvariant.output_data->dimensions[Z];
        } else {
            // Fathom can split an output among invariants but weights don't need to be adjusted
            // The blob has the correct offsets already
            // TODO: What about Fathom blobs which are really SOH/SOK. Is the above logic sufficient?
            nnLog(MVLOG_WARN, "Invariant Z dim different than parent but no slices to broadcast");
        }
    }

    // Point into the 16 byte weight table entry, corresponding to the output channels subset
    variant.weight_table_offset = variant.weight_table_offset << 4;
    variant.output_sparsity_offset = invariant.output_sparsity_offset + srcInvariant.odu_offset;
    int32_t bpp =
        round_up<8>(getODUDTypeSizeBits(convertOutputDtype(srcInvariant.output_data->data_dtype))) >> 3;

    op_size_y = SetupVariant_SOH(invariant, variant);
    invariant.output_sparsity_offset += op_size_y * op_size_x * op_size_z * bpp;

    SetupVariant_NTHW_NTK(variant.registers);

    auto &wt_tensor_ref = srcInvariant.weights_data;

    variant.registers.offset_addr.offset_addr_bf.swizzle_key = srcInvariant.input_data->swizzling_key;
    variant.registers.offset_addr.offset_addr_bf.wt_swizzle_key = wt_tensor_ref.valid() ? wt_tensor_ref->swizzling_key : 0;
    variant.registers.offset_addr.offset_addr_bf.wt_swizzle_sel = 1;

    return true;
}

bool DPUConfigurator::Setup_Input(DPUInvariantRegisters &registers) {
    const auto &input = srcInvariant.input_data;
    if (!input.valid()) {
        nnLog(MVLOG_ERROR, "Missing input data");
        return false;
    }

    // Input Size
    registers.tensor_size0.tensor_size0_bf.tensor_size_x = input->dimensions[X];
    registers.tensor_size0.tensor_size0_bf.tensor_size_y = srcInvariant.parent_input_tensor->dimensions[Y];
    registers.tensor_size1.tensor_size1_bf.tensor_size_z = srcInvariant.parent_input_tensor->dimensions[Z];

    // NOT USED BY RTL. USED BY MODEL TO SUPPORT LEGACY BEHAVIOUR
    registers.z_config.z_config_bf.addr_format_sel = 1;

    auto amode = input->data_dtype;
    auto dtype = convertInputDtype(amode);

    if (dtype == InputTensorDType::UNKNOWN)
        return false;

    registers.tensor_mode.tensor_mode_bf.amode = to_underlying(dtype);

    if (amode != DType::FP16) {
        registers.mpe_cfg.mpe_cfg_bf.mpe_actbias =
            input->quant_zero.size() ? input->quant_zero[0] : 0;

        registers.tensor_mode.tensor_mode_bf.pad_value =
            input->quant_zero.size() ? input->quant_zero[0] : 0;
    }

    bool is_act_dense = input->data.sparsity_index == DEFAULT_INDEX;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense = is_act_dense;

    return true;
}

bool DPUConfigurator::Setup_Weights(DPUInvariantRegisters &registers) {
    if (opType == DPULayerType::MAXPOOL) {
        registers.tensor_mode.tensor_mode_bf.wmode = to_underlying(InputTensorDType::I8);
        return true;
    }

    const auto weights = srcInvariant.weights_data;
    if (!weights.valid()) {
        nnLog(MVLOG_ERROR, "Missing weights data");
        return false;
    }

    auto wmode = weights->data_dtype;
    auto dtype = convertInputDtype(wmode);

    if (dtype == InputTensorDType::UNKNOWN)
        return false;

    registers.tensor_mode.tensor_mode_bf.wmode = to_underlying(dtype);

    if (wmode == DType::U8)
        registers.mpe_cfg.mpe_cfg_bf.mpe_wtbias =
            weights->quant_zero.size() ? weights->quant_zero[0] : 0;

    if (opType == DPULayerType::AVEPOOL) {
        u32f32 wt_data;
        wt_data.f32 = 1.0f;
        switch (weights->data_dtype) {
            case DType::U8:
            case DType::I8:
                registers.elops_wload.elops_wload_bf.pool_wt_data = ((uint8_t)wt_data.f32 << 8) | (uint8_t)wt_data.f32;
                break;
            case DType::FP16:
                registers.elops_wload.elops_wload_bf.pool_wt_data = f32Tof16(wt_data.f32);
                break;
            case DType::BFP16:
                registers.elops_wload.elops_wload_bf.pool_wt_data = f32_to_b16_conv(wt_data.u32, F32_RND_NEAREST_EVEN, 0);
                break;
            default:
                nnLog(MVLOG_ERROR, "weights dtype %d not supported for AVEPOOL", static_cast<int>(weights->data_dtype));
                break;
        }
    }

    return true;
}

bool DPUConfigurator::Setup_Kernel(DPUInvariantRegisters &registers) {
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.rst_ctxt = 1;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = to_underlying(
        (srcInvariant.mpe_frequent_mode == MPE_Mode::VECTOR) ? MPEGrid::GRID_16x1 : MPEGrid::GRID_4x4);

    if (srcInvariant.kernelW < KERNEL_SIZE_MIN || srcInvariant.kernelW > KERNEL_SIZE_MAX) {
        nnLog(MVLOG_ERROR, "Kernel width %u outside of supported range [%u-%u]", srcInvariant.kernelW,
              KERNEL_SIZE_MIN, KERNEL_SIZE_MAX);

        return false;
    }

    registers.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_x = srcInvariant.kernelW;

    if (srcInvariant.kernelH < KERNEL_SIZE_MIN || srcInvariant.kernelH > KERNEL_SIZE_MAX) {
        nnLog(MVLOG_ERROR, "Kernel height %u outside of supported range [%u-%u]", srcInvariant.kernelH,
              KERNEL_SIZE_MIN, KERNEL_SIZE_MAX);

        return false;
    }

    registers.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_y = srcInvariant.kernelH;

    if (srcInvariant.kernel_strideW < KERNEL_STRIDE_MIN || srcInvariant.kernel_strideW > KERNEL_STRIDE_MAX) {
        nnLog(MVLOG_ERROR, "Kernel W stride %u outside of supported range [%u-%u]", srcInvariant.kernel_strideW,
              KERNEL_STRIDE_MIN, KERNEL_STRIDE_MAX);

        return false;
    }

    if (srcInvariant.kernel_strideH < KERNEL_STRIDE_MIN || srcInvariant.kernel_strideH > KERNEL_STRIDE_MAX) {
        nnLog(MVLOG_ERROR, "Kernel H stride %u outside of supported range [%u-%u]", srcInvariant.kernel_strideH,
              KERNEL_STRIDE_MIN, KERNEL_STRIDE_MAX);

        return false;
    }

    registers.tensor_mode.tensor_mode_bf.stride = srcInvariant.kernel_strideW - 1;

    if (srcInvariant.kernel_strideH != srcInvariant.kernel_strideW)
    {
        registers.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y_en = 1;
        registers.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y = srcInvariant.kernel_strideH - 1;
    }

    // When the activations and weights are of different types,
    // MPE_MODE must be configured to the larger of the 2 data types.
    registers.mpe_cfg.mpe_cfg_bf.mpe_mode = std::min(registers.tensor_mode.tensor_mode_bf.amode,
                                                                           registers.tensor_mode.tensor_mode_bf.wmode);

    return true;
}

bool DPUConfigurator::Setup_Output(DPUInvariantRegisters &invariant) {
    const auto output = srcInvariant.output_data;
    if (!output.valid()) {
        nnLog(MVLOG_ERROR, "Missing output data");
        return false;
    }

    bool is_out_dense = output->data.sparsity_index == DEFAULT_INDEX;

    // TODO: This is an estimate based on what's done above for KMB. Nothing in the POC runtime that sets
    // this, so setting to maximum values for now.
    // invariant.odu_be_size = invariant.odu_be_cnt = 2047; // max
    invariant.odu_be_size = invariant.odu_be_cnt = 0;

    // ODU SEs size calculated from output z dimension for 2.7
    invariant.se_size = 0;

    auto dtype = convertOutputDtype(output->data_dtype);
    if (dtype == OutputTensorDType::UNKNOWN)
        return false;

    invariant.odu_cfg.odu_cfg_bf.dtype = to_underlying(dtype);
    invariant.odu_cfg.odu_cfg_bf.mode = 0; // FIXME: how to handle if superdense ?

    SetupInvariant_Grid(invariant);

    invariant.odu_cfg.odu_cfg_bf.write_ac = 1;              // Always write data out!
    invariant.odu_cfg.odu_cfg_bf.write_pt = !is_out_dense;  // Enable/Disable output SE table generation
    invariant.odu_cfg.odu_cfg_bf.write_sp = !is_out_dense;  // Enable/Disable output sparsity map generation
    invariant.odu_cfg.odu_cfg_bf.sp_out_en = !is_out_dense; // Enable/Disable compression of output activations

    invariant.odu_cfg.odu_cfg_bf.swizzle_key = srcInvariant.output_data->swizzling_key;

    invariant.odu_cfg.odu_cfg_bf.permutation = to_underlying(srcInvariant.odu_permutation);

    invariant.odu_cfg.odu_cfg_bf.sp_value = is_out_dense ? 0 : output->quant_zero[0];

    invariant.te_dim1.te_dim1_bf.te_dim_x = output->dimensions[X] - 1;
    invariant.te_dim0.te_dim0_bf.te_dim_y = output->dimensions[Y] - 1;

    // TODO: Why isn't this simply "output->dimensions[Z] - 1" ?
    // TODO: For channel-major output this seems to be incorrect

    {
        auto stride_x = output->strides[STRIDES(X)];
        auto stride_z = output->strides[STRIDES(Z)];
        nnLog(MVLOG_DEBUG, "stride_x = %.3f stride_z = %.3f", stride_x, stride_z);

        if (!stride_z) {
            nnLog(MVLOG_ERROR, "Output stride in z-dimension is zero");
            return false;
        }

        invariant.te_dim0.te_dim0_bf.te_dim_z = (stride_x / stride_z) - 1;
    }

    // Sparse output split over H
    Setup_Output_SOH(invariant, is_out_dense);

    invariant.base_adr[0] = srcInvariant.odu_offset;

    return true;
}

#if 0
bool DPUConfigurator::SetupInvariant_RelativeAddresses(DPUAddresses &addresses) {
    if (const auto *input = srcInvariant.input_data())
        if (!transform(*input, addresses.input_))
            return false;

    if (const auto *output = srcInvariant.output_data())
        if (!transform(*output, addresses.output_))
            return false;

    if (const auto *wt = srcInvariant.weights_table())
        if (!transform(*wt, addresses.weightsTable_))
            return false;

    if (opType_ == nn::dpu_runtime::DPULayerType::MAXPOOL)
        addresses.weights_ = addresses.input_;
    else if (const auto *wd = srcInvariant.weights_data())
        if (!transform(*wd, addresses.weights_))
            return false;

    // FIXME: TBD if MTL HW supports this
    // if (const auto *ppe_task = srcInvariant.ppe_task())
    //     if (const auto *il = ppe_task->instruction_list_data())
    //         if (!transform(*il, addresses.ppe_list_)) return false;

    return true;
}
#endif

unsigned int DPUConfigurator::ConfigWorkloadSize(unsigned int size) const {
    switch (opType) {
        case DPULayerType::CONV:
        case DPULayerType::ELTWISE:
        case DPULayerType::CMCONV:
            // TODO: There seems to be some kind of untold convention with
            // the compiler that this value will be overwritten in runtime
            if (size != srcInvariant.input_data->dimensions[Z])
                nnLog(MVLOG_DEBUG, "Op type %u does not support Z tiling. Got Zsize %u, using %u", static_cast<int>(opType), size,
                      srcInvariant.input_data->dimensions[Z]);

            size = srcInvariant.input_data->dimensions[Z];
            break;

        default:
            break;
    }

    return size;
}

unsigned int DPUConfigurator::ConfigWorkloadStart(unsigned int start) const {
    switch (opType) {
        case DPULayerType::CONV:
        case DPULayerType::ELTWISE:
        case DPULayerType::CMCONV:
            // TODO: There seems to be some kind of untold convention with
            // the compiler that this value will be overwritten in runtime
            if (start != 0)
                nnLog(MVLOG_DEBUG, "Op type %u does not support Z tiling. Got Zstart %u, using 0", static_cast<int>(opType), start);

            start = 0;
            break;

        default:
            break;
    }

    return start;
}

#if 0
// TODO more relocations?
bool Update_Invariant(DPULayerTypes opType, DPUInvariant &invariant, const DPUAddresses &addresses,
                      const NNRelocationData &relocationData) {
    // Hardcoding to 3 for now - matches POC.
    // FIXME: update this for MTL.
    // Assuming 3 is from:
    /*
     * For MeteorLake integration, three contexts can execute simultaneously
     * (two compute engines, each on separate NCE Slices and a copy function on a third context).
     * -- VPU2.7 HAS
     */
    constexpr unsigned int numSlices = 3;
    auto input = addresses.input_;
    Update_Invariant_SOH(opType, invariant, input, relocationData);

    auto adrInput = input.resolve32(relocationData);
    assert((adrInput != 0) && "can't read input from nullptr");

    invariant.registers.act_offset[0] = adrInput;
    invariant.registers.act_offset[1] = adrInput;
    invariant.registers.act_offset[2] = adrInput;
    invariant.registers.act_offset[3] = adrInput;

    invariant.registers.se_sp_addr[1].se_addr = ((1 * SLICE_LENGTH) >> 4);
    ;
    invariant.registers.se_sp_addr[2].se_addr = ((2 * SLICE_LENGTH) >> 4);
    ;
    invariant.registers.se_sp_addr[3].se_addr = ((3 * SLICE_LENGTH) >> 4);
    ;

    // FIXME: hardcoded and directly copied from POC runtime...
    invariant.registers.base_offset_a = 0x200;
    invariant.registers.base_offset_b = 0x602;

    if (!invariant.registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense) {
        invariant.registers.se_sp_addr[0].se_addr =
            input.resolve32(relocationData, RelativeAddress::Class::SparsityTable);
        invariant.registers.se_sp_addr[0].sparsity_addr =
            input.resolve32(relocationData, RelativeAddress::Class::SparsityMap);
    }

    auto adrOutput = addresses.output_.resolve32(relocationData);
    verify_adrOutput(adrOutput.addr(), invariant);

    unsigned int offs[3] = {SLICE_LENGTH >> 4, SLICE_LENGTH >> 4,
                            SLICE_LENGTH >> 4}; // 1024 * 1024 >> 4 as HW requirement
    unsigned int base = RelativeAddress::to_dpu_multicast(adrOutput, offs[0], offs[1], offs[2]);

    if (base < invariant.registers.base_adr[0]) {
        // TODO: Is this a real error? Can it happen?
        nnLog(MVLOG_WARN, "Odu offset %u too large compared to base %u", invariant.registers.base_adr[0], base);
    }

    invariant.registers.base_adr[0] = base - invariant.registers.base_adr[0];

    for (unsigned int i = 0; i < numSlices; ++i) {
        invariant.registers.base_adr[i + 1] = invariant.registers.base_adr[0];

        invariant.registers.odu_cast[i].odu_cast_bf.cast_enable = offs[i] != 0;
        invariant.registers.odu_cast[i].odu_cast_bf.cast_offset = offs[i];
    }

    if (invariant.registers.odu_cfg.odu_cfg_bf.write_pt) {
        auto se_addr = addresses.output_.resolve32(relocationData, RelativeAddress::Class::SparsityTable);
        invariant.registers.pt_base = RelativeAddress::to_dpu_multicast_base(se_addr);
    }

    if (invariant.registers.odu_cfg.odu_cfg_bf.write_sp) {
        auto sp_addr = addresses.output_.resolve32(relocationData, RelativeAddress::Class::SparsityMap);
        invariant.registers.sp_base = RelativeAddress::to_dpu_multicast_base(sp_addr);
    }

    switch (opType) {
        case DPULayerType::CONV:
        case DPULayerType::CMCONV:
        case DPULayerType::MAXPOOL:
        case DPULayerType::AVEPOOL:
        case DPULayerType::DWCONV: {
            RelativeAddress::Class weightClass =
#ifdef WORKAROUND_FORCE_WEIGHTS_OFFSET_TO_NNCMX_BASE_ADR
                RelativeAddress::Class::Base;
#else
                RelativeAddress::Class::Data;
#endif

            auto adrWeights = addresses.weights_.resolve32(relocationData, weightClass);
            assert((adrWeights != 0 || opType == DPULayerType::MAXPOOL || opType == DPULayerType::AVEPOOL) && "can't read Weights from nullptr");

            invariant.registers.wt_offset = adrWeights;

            auto adrWeightsTable = addresses.weightsTable_.resolve32(relocationData);
            assert((adrWeightsTable != 0 || opType == DPULayerType::MAXPOOL || opType == DPULayerType::AVEPOOL) && "can't read WeightsTable from nullptr");

            invariant.registers.weight_start = adrWeightsTable;

            switch (opType) {
                case DPULayerType::DWCONV:
                case DPULayerType::CMCONV:
                case DPULayerType::MAXPOOL:
                    invariant.registers.tensor_start = 0;
                    break;

                default:
                    break;
            }

            break;
        }

        case nn::dpu_runtime::DPULayerType::ELTWISE: {
            auto adrWeights = addresses.weights_.resolve32(relocationData);
            assert((adrWeights != 0) && "can't read Weights from nullptr");

            if (!invariant.registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense) {
                invariant.registers.elop_se_addr =
                    addresses.weights_.resolve32(relocationData, RelativeAddress::Class::SparsityTable);
                invariant.registers.elop_sparsity_addr =
                    addresses.weights_.resolve32(relocationData, RelativeAddress::Class::SparsityMap);
            }
            // Dense Elops
            // Start of Tensor A = adr0_offset[31:0] + [tensor_start[19:0],0000]
            // Start of Tensor B = adr0_offset[31:0] + [weight_start[19:0],0000]
            invariant.registers.act_offset[0] = std::min(adrInput, adrWeights);
            invariant.registers.weight_start =
                (std::max(adrInput, adrWeights) - invariant.registers.act_offset[0]) >> 4;

            break;
        }

        default:
            assert(false && "Layer Type not supported");
            break;
    }

// (EISW-14126, PR#6646): Disable FP16 WL shadowing to avoid DPU hang.
// Commented because not sure if needed. Will be reminder in event we see FP16 hangs.
//     if (addresses.ppe_list_.isValid())
//     {
//         if (auto adr = addresses.ppe_list_.resolve32(relocationData, RelativeAddress::Class::Data, &overflow).addr32())
//         {
//             if (math::round_down<sizeof(unsigned int)>(adr) == math::round_up<sizeof(unsigned int)>(adr))
//                 invariant.ppe_list_ = reinterpret_cast<const unsigned int *>(adr);
//             else
//                 nnLog(MVLOG_WARN, "Unaligned PPE instruction list at %p. Ignoring", adr);
//         }
//     }
//     invariant.allow_shadowing_ =
// #ifndef NN_ENABLE_SPARSE_IDU_SHADOWING
//         invariant.registers.odu_cfg.odu_cfg_bf.ac_dense_mode &&
// #endif
//         // 1) https://hsdes.intel.com/appstore/article/#/18012316304 - FP16 workload shadowing halts for DW (and ZM) ops with odu multicast
//         // 2) https://jira.devtools.intel.com/browse/EISW-14126 - FP16 workload shadowing halts for DW ops w/o odu multicast
//         invariant.registers.odu_cfg.odu_cfg_bf.dtype != static_cast<uint8_t>(OutputTensorDType::FP16);

    return true;
}
#endif

void DPUConfigurator::SetupVariant_NTHW_NTK(DPUVariantRegisters &variant) {
    auto mpe_frequent_mode = srcInvariant.mpe_frequent_mode;

    // Sets up on NTHW on IDU
    nnLog(MVLOG_DEBUG, "mpe_frequent_mode %u", to_underlying(mpe_frequent_mode));
    switch (mpe_frequent_mode) {
        case MPE_Mode::VECTOR:
            variant.offset_addr.offset_addr_bf.nthw_ntk = to_underlying(IDUNthw_Ntk::IDU_8_8);
            break;
        case MPE_Mode::CUBOID_4x16: // NTH = 1, NTW=4, NTK = 16 (4, 16)
            variant.offset_addr.offset_addr_bf.nthw_ntk = to_underlying(IDUNthw_Ntk::IDU_4_16);
            break;
        case MPE_Mode::CUBOID_8x16: // NTH = 2, NTW=4, NTK = 8 (8, 8)
            variant.offset_addr.offset_addr_bf.nthw_ntk = to_underlying(IDUNthw_Ntk::IDU_8_8);
            break;
        case MPE_Mode::CUBOID_16x16: // NTH = 4, NTW=4, NTK = 4  (16, 4)
            variant.offset_addr.offset_addr_bf.nthw_ntk = to_underlying(IDUNthw_Ntk::IDU_16_4);
            break;
        default:
            nnLog(MVLOG_ERROR, "mpe_frequent_mode %u", to_underlying(mpe_frequent_mode));
            break;
    }
}

void DPUConfigurator::SetupInvariant_Grid(DPUInvariantRegisters &invariant) {
    auto mpe_frequent_mode = srcInvariant.mpe_frequent_mode;

    // Sets up on NTHW on IDU
    nnLog(MVLOG_DEBUG, "mpe_frequent_mode %u", to_underlying(mpe_frequent_mode));
    switch (mpe_frequent_mode) {
        case MPE_Mode::VECTOR:
            invariant.odu_cfg.odu_cfg_bf.grid = to_underlying(ODUGrid::GRID_16x1);
            invariant.odu_cfg.odu_cfg_bf.nthw = to_underlying(ODUNthw::NTHW_1);
            invariant.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = to_underlying(MPEGrid::GRID_16x1);
            break;
        case MPE_Mode::CUBOID_4x16: // NTH = 1, NTW=4, NTK = 16 (4, 16)
            invariant.odu_cfg.odu_cfg_bf.grid = to_underlying(ODUGrid::GRID_4x4);
            invariant.odu_cfg.odu_cfg_bf.nthw = to_underlying(ODUNthw::NTHW_4);
            invariant.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = to_underlying(MPEGrid::GRID_4x4);
            break;
        case MPE_Mode::CUBOID_8x16: // NTH = 2, NTW=4, NTK = 8 (8, 8)
            invariant.odu_cfg.odu_cfg_bf.grid = to_underlying(ODUGrid::GRID_4x4);
            invariant.odu_cfg.odu_cfg_bf.nthw = to_underlying(ODUNthw::NTHW_8);
            invariant.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = to_underlying(MPEGrid::GRID_4x4);
            break;
        case MPE_Mode::CUBOID_16x16: // NTH = 4, NTW=4, NTK = 4  (16, 4)
            invariant.odu_cfg.odu_cfg_bf.grid = to_underlying(ODUGrid::GRID_4x4);
            invariant.odu_cfg.odu_cfg_bf.nthw = to_underlying(ODUNthw::NTHW_16);
            invariant.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = to_underlying(MPEGrid::GRID_4x4);
            break;
        default:
            nnLog(MVLOG_ERROR, "mpe_frequent_mode %u", to_underlying(mpe_frequent_mode));
            break;
    }
}

} // namespace parsing_lib

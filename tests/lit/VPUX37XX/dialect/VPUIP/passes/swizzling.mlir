//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --swizzling %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @SetSwizzlingForDpuToDpuBuffer(%in : memref<1x16x56x56xf16, #NHWC, @DDR>,
                        %weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>,
                        %weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
                        -> memref<1x16x56x56xf16, #NHWC, @DDR> {

    %buf0 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>

    %0 = VPUIP.Copy
            inputs(%in : memref<1x16x56x56xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>

    %1 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "CONV"
        }
        input(%0 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%0 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        parent_output(%buf1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        outputs(%buf1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %2 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "CONV"
        }
        input(%1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        parent_output(%buf2 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        outputs(%buf2 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %3 = VPUIP.Copy
            inputs(%2 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
            outputs(%buf3 : memref<1x16x56x56xf16, #NHWC, @DDR>)
             -> memref<1x16x56x56xf16, #NHWC, @DDR>

    return %3 : memref<1x16x56x56xf16, #NHWC, @DDR>

    // Verify that alignment is set only for DPU to DPU buffer

    // CHECK:      [[BUF0:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF1:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
    // CHECK:      [[BUF2:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF3:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @DoNotSetSwizzlingDueToCmxUsageIncrease(%in : memref<1x16x176x175xf16, #NHWC, @DDR>,
                        %weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>,
                        %act_wind : memref<16x1x1x16xui8, #NHWC, @CMX_NN>)
                        -> memref<1x16x176x175xf16, #NHWC, @DDR> {

    %buf0 = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @DDR>

    %0 = VPUIP.Copy
            inputs(%in : memref<1x16x176x175xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x176x175xf16, #NHWC, @CMX_NN>

    %1 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%0 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        activation_window(%act_wind : memref<16x1x1x16xui8, #NHWC, @CMX_NN>)
        parent_input(%0 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
        parent_output(%buf1 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
        outputs(%buf1 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>) -> memref<1x16x176x175xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %2 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%1 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        activation_window(%act_wind : memref<16x1x1x16xui8, #NHWC, @CMX_NN>)
        parent_input(%1 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
        parent_output(%buf2 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
        outputs(%buf2 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>) -> memref<1x16x176x175xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %3 = VPUIP.Copy
            inputs(%2 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
            outputs(%buf3 : memref<1x16x176x175xf16, #NHWC, @DDR>)
             -> memref<1x16x176x175xf16, #NHWC, @DDR>

    return %3 : memref<1x16x176x175xf16, #NHWC, @DDR>

    // Verify that no swizzling is enabled

    // CHECK:      [[BUF0:%.+]] = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @CMX_NN>
    // CHECK-NOT:  VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK:      [[BUF1:%.+]] = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF2:%.+]] = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF3:%.+]] = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @DDR>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x56x56xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<16x1x1x4xsi32, #NHWC, @DDR>
!Weights_DDR = type memref<16x16x1x1xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x56x56xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = type memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = type memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>

func @SetSwizzlingForDpuToDpuBufferInMultiCluster(%input : !Input_DDR,
                        %weights_table : !WeightsTable_DDR,
                        %weights : !Weights_DDR)
                        -> !Output_DDR {

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_1_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output_buff_2_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !Input_DDR) outputs(%input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
    }

    %2 = VPUIP.NCEClusterTiling inputs(%weights_table as %arg0: !WeightsTable_DDR) outputs(%weights_table_cmx as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    }

    %3 = VPUIP.NCEClusterTiling inputs(%weights as %arg0: !Weights_DDR) outputs(%weights_cmx as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
    }

    %4 = VPUIP.NCEClusterTiling
            inputs(%1 as %arg0: !InputStub_CMX,
                    %2 as %arg1: !WeightsTableStub_CMX,
                    %3 as %arg2: !WeightsStub_CMX)
            outputs(%output_buff_1_cmx as %arg3: !OutputStub_CMX)
                -> !OutputStub_CMX {
        %0 = VPUIP.NCEClusterTask
            {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%arg0 : !InputStub_CMX)
            weights(%arg2 : !WeightsStub_CMX)
            weight_table(%arg1 : !WeightsTableStub_CMX)
            parent_input(%arg0 : !InputStub_CMX)
            parent_output(%arg3 : !OutputStub_CMX)
            outputs(%arg3 : !OutputStub_CMX) -> !OutputStub_CMX
            variants :
            {
                DPUTask
                    {
                        outEnd = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                        outStart = [0, 0, 0]
                    }
            }
            PPE :
            {
            }
    }

    %5 = VPUIP.NCEClusterTiling
            inputs(%4 as %arg0: !InputStub_CMX,
                    %2 as %arg1: !WeightsTableStub_CMX,
                    %3 as %arg2: !WeightsStub_CMX)
            outputs(%output_buff_2_cmx as %arg3: !OutputStub_CMX)
                -> !OutputStub_CMX {
        %0 = VPUIP.NCEClusterTask
            {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%arg0 : !InputStub_CMX)
            weights(%arg2 : !WeightsStub_CMX)
            weight_table(%arg1 : !WeightsTableStub_CMX)
            parent_input(%arg0 : !InputStub_CMX)
            parent_output(%arg3 : !OutputStub_CMX)
            outputs(%arg3 : !OutputStub_CMX) -> !OutputStub_CMX
            variants :
            {
                DPUTask
                    {
                        outEnd = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                        outStart = [0, 0, 0]
                    }
            }
            PPE :
            {
            }
    }

    %6 = VPUIP.NCEClusterTiling inputs(%5 as %arg0: !OutputStub_CMX) outputs(%output as %arg1: !Output_DDR) -> !Output_DDR {
        %0 = VPUIP.Copy { out_mem_space = @DDR } inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
    }

    return %6 : !Output_DDR

    // Verify that alignment is set only for DPU to DPU buffer

    // CHECK:      [[BUF_INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_WT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_W:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_1_CMX_SWIZZLED:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN,  {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>
    // CHECK:      [[NCE_CT_COPY_INPUT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:      [[NCE_CT_COPY_WEIGHTS:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:      [[NCE_CT_COPY_ACT_WIN:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:      [[NCE_CT_NCE_TASK:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      outputs([[BUF_OUT_1_CMX_SWIZZLED]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    32x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x56x56xf16, #NHWC, @DDR>
!Weights_DDR = type memref<32x32x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<32x1x1x4xsi32, @DDR>
!Output_DDR = type memref<1x16x56x56xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = type memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = type memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @SetSwizzlingForConstantsMultiClusterDuplicated
func @SetSwizzlingForConstantsMultiClusterDuplicated(%input : !Input_DDR) -> !Output_DDR 
{
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<32x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_1_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output_buff_2_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !Input_DDR) outputs(%input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
    }

    %2 = VPUIP.NCEClusterTiling inputs(%weight_table as %arg0: !WeightsTable_DDR) outputs(%weights_table_cmx as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    }

    %3 = VPUIP.NCEClusterTiling inputs(%weights as %arg0: !Weights_DDR) outputs(%weights_cmx as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
    }

    %4 = VPUIP.NCEClusterTiling
            inputs(%1 as %arg0: !InputStub_CMX,
                    %2 as %arg1: !WeightsTableStub_CMX,
                    %3 as %arg2: !WeightsStub_CMX)
            outputs(%output_buff_1_cmx as %arg3: !OutputStub_CMX) -> !OutputStub_CMX {
        %0 = VPUIP.NCEClusterTask
            {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%arg0 : !InputStub_CMX)
            weights(%arg2 : !WeightsStub_CMX)
            weight_table(%arg1 : !WeightsTableStub_CMX)
            parent_input(%arg0 : !InputStub_CMX)
            parent_output(%arg3 : !OutputStub_CMX)
            outputs(%arg3 : !OutputStub_CMX) -> !OutputStub_CMX
            variants :
            {
                DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    outStart = [0, 0, 0]
                }
            }
            PPE :
            {
            }
        }

    %5 = VPUIP.NCEClusterTiling inputs(%4 as %arg0: !OutputStub_CMX) outputs(%output as %arg1: !Output_DDR) -> !Output_DDR {
        %0 = VPUIP.Copy { out_mem_space = @DDR } inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
    }

    return %5 : !Output_DDR

    // Verify that alignment is set for constants which satisfy the 512 size requirement

    // CHECK:       [[WEIGHT_TABLE:%.+]] = const.Declare memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1> : tensor<32x1x1x4xsi32>, [#const.SwizzleConstant<5 : i64, 3 : i64>]
    // CHECK:       [[WEIGHT:%.+]] = const.Declare memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.SwizzleConstant<5 : i64, 3 : i64>]

    // CHECK:      [[BUF_INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN,  {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN,  {mode = "DUPLICATED", num_clusters = 2 : i64}>
    
    // CHECK:      [[BUF_OUT_1_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>
    
    // CHECK:      [[NCE_CT_COPY_INPUT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:      [[NCE_CT_COPY_WT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:      [[NCE_CT_COPY_WEIGHTS:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:      [[NCE_CT_NCE_TASK:%.+]] = VPUIP.NCEClusterTiling
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    64x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x64x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]  
}>

!Input_DDR = type memref<1x32x3x3xf16, #NHWC, @DDR>
!Weights_DDR = type memref<64x32x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<64x1x1x4xsi32, @DDR>
!Output_DDR = type memref<1x64x3x3xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = type memref<64x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<64x1x1x4xsi32, [@CMX_NN, 0]>
!OutputStub_CMX = type memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @SetSwizzlingForConstantsSOK
func @SetSwizzlingForConstantsSOK(%input : !Input_DDR) -> !Output_DDR 
{
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<64x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !Input_DDR) outputs(%input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
    }

    %2 = VPUIP.NCEClusterTiling inputs(%weight_table as %arg0: !WeightsTable_DDR) outputs(%weights_table_cmx as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    }

    %3 = VPUIP.NCEClusterTiling inputs(%weights as %arg0: !Weights_DDR) outputs(%weights_cmx as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
    }

    %4 = VPUIP.NCEClusterTiling
            inputs(%1 as %arg0: !InputStub_CMX,
                    %2 as %arg1: !WeightsTableStub_CMX,
                    %3 as %arg2: !WeightsStub_CMX)
            outputs(%output_buff_cmx as %arg3: !OutputStub_CMX) -> !OutputStub_CMX {
        %0 = VPUIP.NCEClusterTask
            {
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
                kernel_size = [1, 1], 
                kernel_strides = [1, 1], 
                minimumHardwareExecutionCost = 177 : i64, 
                task_type = "CONV"
            }
            input(%arg0 : !InputStub_CMX)
            weights(%arg2 : !WeightsStub_CMX)
            weight_table(%arg1 : !WeightsTableStub_CMX)
            parent_input(%arg0 : !InputStub_CMX)
            parent_output(%arg3 : !OutputStub_CMX)
            outputs(%arg3 : !OutputStub_CMX) -> !OutputStub_CMX
            variants :
            {
                DPUTask {
                    cluster_id = 0 : i64, outEnd = [2, 2, 63], 
                    mpe_mode = "CUBOID_16x16", 
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
                    outStart = [0, 0, 0]}
            }
            PPE :
            {
                PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        }

    %5 = VPUIP.NCEClusterTiling inputs(%4 as %arg0: !OutputStub_CMX) outputs(%output as %arg1: !Output_DDR) -> !Output_DDR {
        %0 = VPUIP.Copy { out_mem_space = @DDR } inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
    }

    return %5 : !Output_DDR

    // Verify that alignment is set for constants which satisfy the 512 size requirement

    // CHECK:       [[CST_WT:%.+]] = const.Declare memref<64x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    // CHECK-SAME:      const.SwizzleConstant<5 : i64, 3 : i64>
    // CHECK:       [[CST_W:%.+]] = const.Declare memref<64x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    // CHECK-SAME:      const.SwizzleConstant<5 : i64, 3 : i64>

    // CHECK:   [[BUF_INPUT:%.+]] = VPURT.AllocDistributed
    // CHECK:   [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK-SAME:   VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}
    // CHECK:   [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK-SAME:   VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}
    // CHECK:   [[BUF_OUT_CMX:%.+]] = VPURT.AllocDistributed
}

// -----

!qElemType0 = type !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:1, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127,1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127,1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType2 = type !quant.uniform<u8<0:254>:f16:0, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127}>
!qElemType3 = type !quant.uniform<u8<0:254>:f16:0, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127,1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127,1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType4 = type !quant.uniform<u8<0:254>:f16:0, {1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType5 = type !quant.uniform<u8<0:254>:f16:1, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127}>
!qElemType6 = type !quant.uniform<u8<0:254>:f16:1, {1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType0, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    64x16x1x1x!qElemType3, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x32x32x!qElemType0, #NHWC, @DDR>
!Output_DDR = type memref<1x64x32x32x!qElemType1, #NHWC, @DDR>
!Weights_DDR = type memref<64x16x1x1x!qElemType3, #NHWC, @DDR>
!WeightsTable_DDR = type memref<64x1x1x4xsi32, #NCHW, @DDR>

!InputStub_CMX = type memref<1x16x32x32x!qElemType0, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x64x32x32x!qElemType1, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<64x16x1x1x!qElemType3, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<64x1x1x4xsi32, #NCHW, @CMX_NN>

func @SetSwizzlingForQuantConstantsSOK(%input : !Input_DDR) -> !Output_DDR 
{
    %weights = const.Declare memref<64x16x1x1x!qElemType3, #NHWC> =
        dense<1.0> : tensor<64x16x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType3>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !Input_DDR) outputs(%input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
    }

    %2 = VPUIP.NCEClusterTiling inputs(%weights_table as %arg0: !WeightsTable_DDR) outputs(%weights_table_cmx as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    }

    %3 = VPUIP.NCEClusterTiling inputs(%weights as %arg0: !Weights_DDR) outputs(%weights_cmx as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
    }

    %4 = VPUIP.NCEClusterTiling
            inputs(%1 as %arg0: !InputStub_CMX,
                    %2 as %arg1: !WeightsTableStub_CMX,
                    %3 as %arg2: !WeightsStub_CMX)
            outputs(%output_buff_cmx as %arg3: !OutputStub_CMX) -> !OutputStub_CMX {
        %0 = VPUIP.NCEClusterTask
            {
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
                kernel_size = [1, 1], 
                kernel_strides = [1, 1], 
                minimumHardwareExecutionCost = 177 : i64, 
                task_type = "CONV"
            }
            input(%arg0 : !InputStub_CMX)
            weights(%arg2 : !WeightsStub_CMX)
            weight_table(%arg1 : !WeightsTableStub_CMX)
            parent_input(%arg0 : !InputStub_CMX)
            parent_output(%arg3 : !OutputStub_CMX)
            outputs(%arg3 : !OutputStub_CMX) -> !OutputStub_CMX
            variants :
            {
                DPUTask {
                    cluster_id = 0 : i64, outEnd = [2, 2, 63], 
                    mpe_mode = "CUBOID_16x16", 
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
                    outStart = [0, 0, 0]}
            }
            PPE :
            {
                PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        }

    %5 = VPUIP.NCEClusterTiling inputs(%4 as %arg0: !OutputStub_CMX) outputs(%output as %arg1: !Output_DDR) -> !Output_DDR {
        %0 = VPUIP.Copy { out_mem_space = @DDR } inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
    }

    return %5 : !Output_DDR

    // Verify that alignment is set for constants which satisfy the 512 size requirement

    // CHECK:   [[BUF_INPUT:%.+]] = VPURT.AllocDistributed
    // CHECK:   [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK-SAME:   VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}
    // CHECK:   [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK-SAME:   VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}
    // CHECK:   [[BUF_OUT_CMX:%.+]] = VPURT.AllocDistributed
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @DoNotSetSwizzlingDueToActivationWindow(%in : memref<1x16x56x56xf16, #NHWC, @DDR>,
                        %weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>,
                        %act_wind : memref<16x1x1x16xui8, #NHWC, @CMX_NN>)
                        -> memref<1x16x56x56xf16, #NHWC, @DDR> {

    %buf0 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>

    %0 = VPUIP.Copy
            inputs(%in : memref<1x16x56x56xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>

    %1 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%0 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        activation_window(%act_wind : memref<16x1x1x16xui8, #NHWC, @CMX_NN>)
        parent_input(%0 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        parent_output(%buf1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        outputs(%buf1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %2 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        activation_window(%act_wind : memref<16x1x1x16xui8, #NHWC, @CMX_NN>)
        parent_input(%1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        parent_output(%buf2 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        outputs(%buf2 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %3 = VPUIP.Copy
            inputs(%2 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
            outputs(%buf3 : memref<1x16x56x56xf16, #NHWC, @DDR>)
             -> memref<1x16x56x56xf16, #NHWC, @DDR>

    return %3 : memref<1x16x56x56xf16, #NHWC, @DDR>

    // Swizzling is temporarily disabled for operations with activation window

    // CHECK:      [[BUF0:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK-NOT:      [[BUF1:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
    // CHECK:      [[BUF1:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF2:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF3:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>

}

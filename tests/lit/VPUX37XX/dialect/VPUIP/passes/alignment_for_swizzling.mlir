// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --alignment-for-swizzling %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @SetSwizzlingForDpuToDpuBuffer(%in : memref<1x16x56x56xf16, #NHWC, @DDR>,
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
                    end = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    start = [0, 0, 0]
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
                    end = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    start = [0, 0, 0]
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
    // CHECK:      [[BUF1:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<1x16x56x56xf16, {order = #NHWC, swizzlingKey = 5 : i64}, @CMX_NN>
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
                    end = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    start = [0, 0, 0]
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
                    end = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    start = [0, 0, 0]
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

!ActWinDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x16xui8, #NHWC, @CMX_NN, {
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
!ActWin_DDR = type memref<16x1x1x16xui8, #NHWC, @DDR>
!Output_DDR = type memref<1x16x56x56xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!ActWinStub_CMX = type memref<16x1x1x16xui8, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = type memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>

func @SetSwizzlingForDpuToDpuBufferInMultiCluster(%input : !Input_DDR,
                        %weights_table : !WeightsTable_DDR,
                        %act_win : !ActWin_DDR)
                        -> !Output_DDR {

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %act_win_cmx = VPURT.AllocDistributed -> !ActWinDistributed
    %output_buff_1_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output_buff_2_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !Input_DDR) outputs(%input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
    }

    %2 = VPUIP.NCEClusterTiling inputs(%weights_table as %arg0: !WeightsTable_DDR) outputs(%weights_table_cmx as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    }

    %3 = VPUIP.NCEClusterTiling inputs(%act_win as %arg0: !ActWin_DDR) outputs(%act_win_cmx as %arg1: !ActWinStub_CMX) -> !ActWinDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !ActWin_DDR) outputs(%arg1: !ActWinStub_CMX) -> !ActWinStub_CMX
    }

    %4 = VPUIP.NCEClusterTiling
            inputs(%1 as %arg0: !InputStub_CMX,
                    %2 as %arg1: !WeightsTableStub_CMX,
                    %3 as %arg2: !ActWinStub_CMX)
            outputs(%output_buff_1_cmx as %arg3: !OutputStub_CMX)
                -> !OutputStub_CMX {
        %0 = VPUIP.NCEClusterTask
            {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "MAXPOOL"
            }
            input(%arg0 : !InputStub_CMX)
            weight_table(%arg1 : !WeightsTableStub_CMX)
            activation_window(%arg2 : !ActWinStub_CMX)
            parent_input(%arg0 : !InputStub_CMX)
            parent_output(%arg3 : !OutputStub_CMX)
            outputs(%arg3 : !OutputStub_CMX) -> !OutputStub_CMX
            variants :
            {
                DPUTask
                    {
                        end = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                        start = [0, 0, 0]
                    }
            }
            PPE :
            {
            }
    }

    %5 = VPUIP.NCEClusterTiling
            inputs(%4 as %arg0: !InputStub_CMX,
                    %2 as %arg1: !WeightsTableStub_CMX,
                    %3 as %arg2: !ActWinStub_CMX)
            outputs(%output_buff_2_cmx as %arg3: !OutputStub_CMX)
                -> !OutputStub_CMX {
        %0 = VPUIP.NCEClusterTask
            {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "MAXPOOL"
            }
            input(%arg0 : !InputStub_CMX)
            weight_table(%arg1 : !WeightsTableStub_CMX)
            activation_window(%arg2 : !ActWinStub_CMX)
            parent_input(%arg0 : !InputStub_CMX)
            parent_output(%arg3 : !OutputStub_CMX)
            outputs(%arg3 : !OutputStub_CMX) -> !OutputStub_CMX
            variants :
            {
                DPUTask
                    {
                        end = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                        start = [0, 0, 0]
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
    // CHECK:      [[BUF_ACT_WIN:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x16xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_1_CMX_SWIZZLED:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x16x56x56xf16, {order = #NHWC, swizzlingKey = 5 : i64}, @CMX_NN,  {mode = "DUPLICATED", num_clusters = 2 : i64}>
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

!Input_DDR = type memref<1x16x16x16xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x16x16xf16, #NHWC, @DDR>
!Weights_DDR = type memref<32x32x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<16x1x1x4xsi32, @DDR>

!InputStub_CMX = type memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<32x1x1x4xsi32, #NHWC, @CMX_NN>

// CHECK-LABEL: @ResizeWeightTableBuffer
func @ResizeWeightTableBuffer(%in : !Input_DDR) -> !OutputStub_CMX {

    %buf0 = memref.alloc() : !InputStub_CMX
    %buf1 = memref.alloc() : !OutputStub_CMX
    %buf2 = memref.alloc() : !WeightsTableStub_CMX
    %buf3 = memref.alloc() : !WeightsStub_CMX

    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPUIP.Copy inputs(%weight_table : !WeightsTable_DDR) outputs(%buf2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    %1 = VPUIP.Copy inputs(%in : !Input_DDR) outputs(%buf0 : !InputStub_CMX) -> !InputStub_CMX
    %2 = VPUIP.Copy inputs(%weights : !Weights_DDR) outputs(%buf3 : !WeightsStub_CMX) -> !WeightsStub_CMX
    %3 = VPUIP.NCEClusterTask 
        {
            activation_window_channel_length = 27 : i64, 
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
            kernel_size = [1, 1], 
            kernel_strides = [1, 1], 
            task_type = "CONV"
        } 
        input(%1 : !InputStub_CMX) 
        weights(%2 : !WeightsStub_CMX) 
        weight_table(%0 : !WeightsTableStub_CMX) 
        parent_input(%1 : !InputStub_CMX) 
        parent_output(%buf1 : !OutputStub_CMX) 
        outputs(%buf1 : !OutputStub_CMX) -> !OutputStub_CMX
        variants :  
        {
            DPUTask 
                {
                    end = [55, 10, 15], mpe_mode = "VECTOR_FP16", 
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
                    start = [0, 0, 0]
                }
        }   
        PPE :  
        {
        }
    return %3 : !OutputStub_CMX


    // CHECK: [[OUTPUT_MEMREF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK: [[INPUT_MEMREF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK: [[WEIGHTS_BUFFER:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<32x32x1x1xf16, {order = #NHWC, swizzlingKey = 5 : i64}, @CMX_NN>
    // CHECK: [[WEIGHTS:%.+]] = const.Declare memref<32x32x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK: [[WEIGHT_TABLE:%.+]] = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>, [#const.PadWithZero<[0, 0, 0, 0], [16, 0, 0, 0]>]
    // CHECK: [[WEIGHT_TABLE_BUFFER:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<32x1x1x4xsi32, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>
	
    // CHECK: [[WT_IP:%.*]] = VPUIP.Copy inputs([[WEIGHT_TABLE]] : memref<32x1x1x4xsi32>)
    // CHECK-SAME: outputs([[WEIGHT_TABLE_BUFFER]] : memref<32x1x1x4xsi32, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>) -> memref<32x1x1x4xsi32, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>
    // CHECK: [[VAR_INPUT1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME: outputs([[OUTPUT_MEMREF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK: [[W_IP:%.*]] = VPUIP.Copy inputs([[WEIGHTS]] : memref<32x32x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME: outputs([[WEIGHTS_BUFFER]] : memref<32x32x1x1xf16, {order = #NHWC, swizzlingKey = 5 : i64}, @CMX_NN>) -> memref<32x32x1x1xf16, {order = #NHWC, swizzlingKey = 5 : i64}, @CMX_NN>
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
                    end = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    start = [0, 0, 0]
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

    // CHECK:      [[BUF_INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, {order = #NHWC, swizzlingKey = 5 : i64}, @CMX_NN,  {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x32x1x1xf16, {order = #NHWC, swizzlingKey = 5 : i64}, @CMX_NN,  {mode = "DUPLICATED", num_clusters = 2 : i64}>
    
    // CHECK:      [[BUF_OUT_1_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>
    
    // CHECK:      [[NCE_CT_COPY_INPUT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:      [[NCE_CT_COPY_WT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:      [[NCE_CT_COPY_WEIGHTS:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:      [[NCE_CT_NCE_TASK:%.+]] = VPUIP.NCEClusterTiling
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_DDR = type memref<1x16x16x16xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x16x16xf16, #NHWC, @DDR>
!Weights_DDR = type memref<32x32x1x1xf16, #NHWC, @DDR>
!WeightsSM_DDR = type memref<32x1x1x128xi1, @DDR>
!WeightsTable_DDR = type memref<16x1x1x4xsi32, @DDR>

!InputStub_CMX = type memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsSMStub_CMX = type memref<32x1x1x128xi1, @CMX_NN>
!WeightsTableStub_CMX = type memref<32x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @ConvSparseWeights
func @ConvSparseWeights(%in : !Input_DDR) -> !OutputStub_CMX {

    %buf0 = memref.alloc() : !InputStub_CMX
    %buf1 = memref.alloc() : !OutputStub_CMX
    %buf2 = memref.alloc() : !WeightsTableStub_CMX
    %buf3 = memref.alloc() : !WeightsStub_CMX
    %buf4 = memref.alloc() : !WeightsSMStub_CMX

    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    %0 = VPUIP.Copy inputs(%weight_table : !WeightsTable_DDR) outputs(%buf2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    %1 = VPUIP.Copy inputs(%in : !Input_DDR) outputs(%buf0 : !InputStub_CMX) -> !InputStub_CMX
    %2 = VPUIP.Copy inputs(%weights : !Weights_DDR) outputs(%buf3 : !WeightsStub_CMX) -> !WeightsStub_CMX
    %3 = VPUIP.Copy inputs(%weights_sm : !WeightsSM_DDR) outputs(%buf4 : !WeightsSMStub_CMX) -> !WeightsSMStub_CMX

    %4 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "CONV"
        }
        input(%1 : !InputStub_CMX)
        weights(%2 : !WeightsStub_CMX)
        weights_sparsity_map(%3 : !WeightsSMStub_CMX)
        weight_table(%0 : !WeightsTableStub_CMX)
        parent_input(%1 : !InputStub_CMX)
        parent_output(%buf1 : !OutputStub_CMX)
        outputs(%buf1 : !OutputStub_CMX) -> !OutputStub_CMX
        variants : {
            DPUTask { end = [55, 10, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
        }
        PPE : {
        }
    return %4 : !OutputStub_CMX


    // CHECK:       [[OUTPUT_MEMREF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[INPUT_MEMREF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[WEIGHTS_BUFFER:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<32x32x1x1xf16, {order = #NHWC, swizzlingKey = 5 : i64}, @CMX_NN>
    // CHECK:       [[WEIGHTS_SM_BUFFER:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<32x1x1x128xi1, {order = #NCHW, swizzlingKey = 5 : i64}, @CMX_NN>
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare memref<32x32x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK:       [[WEIGHTS_SM:%.+]] = const.Declare memref<32x1x1x128xi1, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:       [[WEIGHT_TABLE:%.+]] = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>, [#const.PadWithZero<[0, 0, 0, 0], [16, 0, 0, 0]>]
    // CHECK:       [[WEIGHT_TABLE_BUFFER:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<32x1x1x4xsi32, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>

    // CHECK:       [[WT:%.*]] = VPUIP.Copy inputs([[WEIGHT_TABLE]] : memref<32x1x1x4xsi32>)
    // CHECK-SAME:                          outputs([[WEIGHT_TABLE_BUFFER]] : memref<32x1x1x4xsi32, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>)
    // CHECK-SAME:                          -> memref<32x1x1x4xsi32, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>
    // CHECK:       [[INPUT:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:                             outputs([[OUTPUT_MEMREF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                             -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[W:%.*]] = VPUIP.Copy inputs([[WEIGHTS]] : memref<32x32x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                         outputs([[WEIGHTS_BUFFER]] : memref<32x32x1x1xf16, {order = #NHWC, swizzlingKey = 5 : i64}, @CMX_NN>)
    // CHECK-SAME:                         -> memref<32x32x1x1xf16, {order = #NHWC, swizzlingKey = 5 : i64}, @CMX_NN>
    // CHECK:       [[W_SM:%.*]] = VPUIP.Copy inputs([[WEIGHTS_SM]] : memref<32x1x1x128xi1, @DDR>)
    // CHECK-SAME:                            outputs([[WEIGHTS_SM_BUFFER]] : memref<32x1x1x128xi1, {order = #NCHW, swizzlingKey = 5 : i64}, @CMX_NN>)
    // CHECK-SAME:                            -> memref<32x1x1x128xi1, {order = #NCHW, swizzlingKey = 5 : i64}, @CMX_NN>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
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

!WeightsSMDistributed = type !VPUIP.DistributedBuffer<
    32x1x1x128xi1, #NCHW, @CMX_NN, {
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
!WeightsSM_DDR = type memref<32x1x1x128xi1, @DDR>
!WeightsTable_DDR = type memref<32x1x1x4xsi32, @DDR>
!Output_DDR = type memref<1x16x56x56xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = type memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsSMStub_CMX = type memref<32x1x1x128xi1, @CMX_NN>
!WeightsTableStub_CMX = type memref<32x1x1x4xsi32, [@CMX_NN, 0]>
!OutputStub_CMX = type memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @ConvSparseWeightsMulticlustering
func @ConvSparseWeightsMulticlustering(%input : !Input_DDR) -> !Output_DDR
{
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<32x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx = VPURT.AllocDistributed -> !WeightsSMDistributed
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

    %4 = VPUIP.NCEClusterTiling inputs(%weights_sm as %arg0: !WeightsSM_DDR) outputs(%weights_sm_cmx as %arg1: !WeightsSMStub_CMX) -> !WeightsSMDistributed {
        %0 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !WeightsSM_DDR) outputs(%arg1: !WeightsSMStub_CMX) -> !WeightsSMStub_CMX
    }

    %5 = VPUIP.NCEClusterTiling
            inputs(%1 as %arg0: !InputStub_CMX,
                    %2 as %arg1: !WeightsTableStub_CMX,
                    %3 as %arg2: !WeightsStub_CMX,
                    %4 as %arg3: !WeightsSMStub_CMX)
            outputs(%output_buff_1_cmx as %arg4: !OutputStub_CMX) -> !OutputStub_CMX {
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
            weights_sparsity_map(%arg3 : !WeightsSMStub_CMX)
            weight_table(%arg1 : !WeightsTableStub_CMX)
            parent_input(%arg0 : !InputStub_CMX)
            parent_output(%arg4 : !OutputStub_CMX)
            outputs(%arg4 : !OutputStub_CMX) -> !OutputStub_CMX
            variants :
            {
                DPUTask
                {
                    end = [55, 55, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    start = [0, 0, 0]
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

    // Verify that alignment is set for constants which satisfy the 512 size requirement

    // CHECK:       [[CST_WEIGHT_TABLE:%.+]] = const.Declare memref<32x1x1x4xsi32, @DDR> = dense<1> : tensor<32x1x1x4xsi32>
    // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare memref<32x32x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK:       [[CST_WEIGHTS_SM:%.+]] = const.Declare memref<32x1x1x128xi1, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:       [[BUF_INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, {order = #NHWC, swizzlingKey = 5 : i64}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x32x1x1xf16, {order = #NHWC, swizzlingKey = 5 : i64}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUF_WEIGHTS_SM:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x128xi1, {order = #NCHW, swizzlingKey = 5 : i64}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[BUF_OUT_1_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>

    // CHECK:       [[COPY_INPUT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0
    // CHECK-SAME:      outputs([[BUF_INPUT]]
    // CHECK:       [[COPY_WT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[CST_WEIGHT_TABLE]]
    // CHECK-SAME:      outputs([[BUF_WT]]
    // CHECK:       [[COPY_WEIGHTS:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[CST_WEIGHTS]]
    // CHECK-SAME:      outputs([[BUF_WEIGHTS]]
    // CHECK:       [[NCE_CT_NCE_TASK:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[CST_WEIGHTS_SM]]
    // CHECK-SAME:      outputs([[BUF_WEIGHTS_SM]]
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
                    cluster_id = 0 : i64, end = [2, 2, 63], 
                    mpe_mode = "CUBOID_16x16", 
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
                    start = [0, 0, 0]}
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
    // CHECK:   [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK:   [[BUF_OUT_CMX:%.+]] = VPURT.AllocDistributed
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    96x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    96x1x1x4xsi32, #NCHW, @CMX_NN, {
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
!Weights_DDR = type memref<96x32x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<96x1x1x4xsi32, @DDR>
!Output_DDR = type memref<1x64x3x3xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = type memref<96x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<96x1x1x4xsi32, [@CMX_NN, 0]>
!OutputStub_CMX = type memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @DoNotSetSwizzlingForConstantsSOK
func @DoNotSetSwizzlingForConstantsSOK(%input : !Input_DDR) -> !Output_DDR 
{
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<96x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<96x32x1x1xf16>, [#const.Reorder<#NHWC>]

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
                    cluster_id = 0 : i64, end = [2, 2, 63], 
                    mpe_mode = "CUBOID_16x16", 
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
                    start = [0, 0, 0]}
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

    // Verify that alignment is not set for constants which do satisfy the 512 size requirement

    // CHECK:   [[BUF_INPUT:%.+]] = VPURT.AllocDistributed
    // CHECK:   [[BUF_WT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<96x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:   [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<96x32x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
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
                    cluster_id = 0 : i64, end = [2, 2, 63], 
                    mpe_mode = "CUBOID_16x16", 
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
                    start = [0, 0, 0]}
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
    // CHECK:   [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK:   [[BUF_OUT_CMX:%.+]] = VPURT.AllocDistributed
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_DDR = type memref<1x16x16x16xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x16x16xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<32x1x1x4xsi32, @CMX_NN>
!WeightsStub_CMX = type memref<32x2x1x4xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @SetSwizzlingForFusedConstant
func @SetSwizzlingForFusedConstant(%input: !Input_DDR) -> !OutputStub_CMX {
    %cst = const.Declare memref<1x1x1x1024xui8> = dense<1> : tensor<1x1x1x1024xui8>
    %0 = memref.alloc() : !InputStub_CMX
    %1 = memref.alloc() : !OutputStub_CMX
    %2 = VPUIP.Copy inputs(%input : !Input_DDR) outputs(%0 : !InputStub_CMX) -> !InputStub_CMX
    %3 = memref.alloc() : memref<1x1x1x1024xui8, [@CMX_NN, 0]>

    %4 = VPUIP.Copy inputs(%cst : memref<1x1x1x1024xui8>) outputs(%3 : memref<1x1x1x1024xui8, [@CMX_NN, 0]>) -> memref<1x1x1x1024xui8, [@CMX_NN, 0]>
    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 1, 1, 512] : memref<1x1x1x1024xui8, [@CMX_NN, 0]> to memref<1x1x1x512xui8, {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]>
    %6 = VPUIP.ViewOp %5 : memref<1x1x1x512xui8, {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]> to !WeightsTableStub_CMX

    %7 = VPUIP.SubView %4 [0, 0, 0, 512] [1, 1, 1, 512] : memref<1x1x1x1024xui8, [@CMX_NN, 0]> to memref<1x1x1x512xui8, {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]>
    %8 = VPUIP.ViewOp %7 : memref<1x1x1x512xui8, {order = #NCHW, strides = [1024, 1024, 1024, 1]}, [@CMX_NN, 0]> to !WeightsStub_CMX

    %9 = VPUIP.NCEClusterTask 
    {
        activation_window_channel_length = 27 : i64, 
        constantsFused = true, 
        kernel_padding = 
        {
            bottom = 0 : i64, 
            left = 0 : i64, 
            right = 0 : i64, 
            top = 0 : i64
        }, 
        kernel_size = [1, 1], 
        kernel_strides = [1, 1], 
        task_type = "CONV"
    } 
    input(%2 : !InputStub_CMX) 
    weights(%8 : !WeightsStub_CMX) 
    weight_table(%6 : !WeightsTableStub_CMX) 
    parent_input(%2 : !InputStub_CMX) 
    parent_output(%1 : !OutputStub_CMX) 
    outputs(%1 : !OutputStub_CMX) -> !OutputStub_CMX 
    variants :  
    {
        DPUTask 
        {
            end = [55, 10, 15], 
            mpe_mode = "VECTOR_FP16", 
            pad = 
            {
                bottom = 0 : i64, 
                left = 0 : i64, 
                right = 0 : i64, 
                top = 0 : i64
            }, 
            start = [0, 0, 0]
        }
    } 
    PPE :  {
    }
    return %9 : !OutputStub_CMX

    // Verify that alignment is set for fused constant which satisfy the 512 size requirement

    // CHECK:       [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x1024xui8>
    // CHECK:       memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAR_INPUT:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs(%0 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAR0:%.*]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<1x1x1x1024xui8, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>
    // CHECK:       [[VAR1:%.*]] = VPUIP.Copy inputs([[FUSED_CONSTANT]] : memref<1x1x1x1024xui8>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x1x1x1024xui8, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>) -> memref<1x1x1x1024xui8, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>

    // CHECK:       [[VAR2:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 0] [1, 1, 1, 512] : 
    // CHECK-SAME:      memref<1x1x1x1024xui8, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]> to memref<1x1x1x512xui8, 
    // CHECK-SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1], swizzlingKey = 5 : i64}, [@CMX_NN, 0]>

    // CHECK:       [[VAR3:%.*]] = VPUIP.ViewOp [[VAR2]] : memref<1x1x1x512xui8, 
    // CHECK_SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1], swizzlingKey = 5 : i64}, [@CMX_NN, 0]> to memref<32x1x1x4xsi32, {order = #NCHW, swizzlingKey = 5 : i64}, @CMX_NN>

    // CHECK:       [[VAR4:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 512] [1, 1, 1, 512] : 
    // CHECK-SAME:      memref<1x1x1x1024xui8, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]> to memref<1x1x1x512xui8, 
    // CHECK-SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1], swizzlingKey = 5 : i64}, [@CMX_NN, 0]>

    // CHECK:       [[VAR5:%.*]] = VPUIP.ViewOp [[VAR4]] : memref<1x1x1x512xui8,
    // CHECK_SAME:      {order = #NCHW, strides = [1024, 1024, 1024, 1], swizzlingKey = 5 : i64}, [@CMX_NN, 0]> to memref<32x2x1x4xf16, {order = #NCHW, swizzlingKey = 5 : i64}, @CMX_NN>

    // CHECK:       [[VAR6:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          activation_window_channel_length = 27 : i64,
    // CHECK-SAME:          constantsFused = true,
    // CHECK-SAME:          kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = "CONV"
    // CHECK-SAME:      input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights([[VAR5]] : memref<32x2x1x4xf16, {order = #NHWC, swizzlingKey = 5 : i64}, @CMX_NN>)
    // CHECK-SAME:      weight_table([[VAR3]] : memref<32x1x1x4xsi32, {order = #NCHW, swizzlingKey = 5 : i64}, @CMX_NN>)
    // CHECK-SAME:      parent_input([[VAR_INPUT]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:           DPUTask
    // CHECK-SAME:          end = [55, 10, 15],
    // CHECK-SAME:          mpe_mode = "VECTOR_FP16",
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          start = [0, 0, 0]

    // CHECK:       return [[VAR6]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
}

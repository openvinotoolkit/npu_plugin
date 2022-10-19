// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --alignment-for-swizzling %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @SetSwizzlingForDpuToDpuBuffer(%in : memref<1x16x56x56xf16, #NHWC, @DDR>,
                        %weight_table : memref<16x1x1x4xsi32, @CMX_NN>,
                        %act_wind : memref<16x1x1x16xui8, @CMX_NN>)
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
        weight_table(%weight_table : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_wind : memref<16x1x1x16xui8, @CMX_NN>)
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
        weight_table(%weight_table : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_wind : memref<16x1x1x16xui8, @CMX_NN>)
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
    // CHECK:      [[BUF1:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF2:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF3:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>

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
    // CHECK:      [[BUF_OUT_1_CMX_SWIZZLED:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
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
!WeightsTableStub_CMX = type memref<32x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @ResizeWeightTableBuffer
func @ResizeWeightTableBuffer(%in : !Input_DDR) -> !OutputStub_CMX {

    %buf0 = memref.alloc() : !InputStub_CMX
    %buf1 = memref.alloc() : !OutputStub_CMX
    %buf2 = memref.alloc() : !WeightsTableStub_CMX
    %buf3 = memref.alloc() : !WeightsStub_CMX

    %weight_table = const.Declare !WeightsTable_DDR = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>
    %weights = const.Declare !Weights_DDR = #const.Content<dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]>

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

    // CHECK: [[WEIGHTS_BUFFER:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<32x32x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[WEIGHTS:%.+]] = const.Declare memref<32x32x1x1xf16, #NHWC, @DDR> = #const.Content<dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]>
    // CHECK: [[WEIGHT_TABLE:%.+]] = const.Declare memref<32x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.PadWithZero<[0, 0, 0, 0], [16, 0, 0, 0]>]>
    // CHECK: [[WEIGHT_TABLE_BUFFER:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
	
    // CHECK: [[WT_IP:%.*]] = VPUIP.Copy inputs([[WEIGHT_TABLE]] : memref<32x1x1x4xsi32>)
    // CHECK-SAME: outputs([[WEIGHT_TABLE_BUFFER]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>       
    // CHECK: [[VAR_INPUT1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME: outputs([[OUTPUT_MEMREF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK: [[W_IP:%.*]] = VPUIP.Copy inputs([[WEIGHTS]] : memref<32x32x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME: outputs([[WEIGHTS_BUFFER]] : memref<32x32x1x1xf16, #NHWC, @CMX_NN>) -> memref<32x32x1x1xf16, #NHWC, @CMX_NN>
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
!WeightsTableStub_CMX = type memref<32x1x1x4xsi32, [@CMX_NN, 0]>
!OutputStub_CMX = type memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @SetSwizzlingForConstantsMultiClusterDuplicated
func @SetSwizzlingForConstantsMultiClusterDuplicated(%input : !Input_DDR) -> !Output_DDR 
{
    %weight_table = const.Declare !WeightsTable_DDR = #const.Content<dense<1> : tensor<32x1x1x4xsi32>>
    %weights = const.Declare !Weights_DDR = #const.Content<dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]>

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
    // CHECK:      [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    
    // CHECK:      [[BUF_OUT_1_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>
    
    // CHECK:      [[NCE_CT_COPY_INPUT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:      [[NCE_CT_COPY_WT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:      [[NCE_CT_COPY_WEIGHTS:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:      [[NCE_CT_NCE_TASK:%.+]] = VPUIP.NCEClusterTiling
}

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swizzling %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_DDR = memref<1x16x16x16xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x16x16xf16, #NHWC, @DDR>
!Weights_DDR = memref<32x32x1x1xf16, #NHWC, @DDR>
!WeightsSM_DDR = memref<32x1x1x128xi1, @DDR>
!WeightsTable_DDR = memref<32x1x1x4xsi32, @DDR>

!InputStub_CMX = memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsSMStub_CMX = memref<32x1x1x128xi1, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @ConvSparseWeights
func.func @ConvSparseWeights(%in : !Input_DDR) -> !OutputStub_CMX {

    %buf0 = memref.alloc() : !InputStub_CMX
    %buf1 = memref.alloc() : !OutputStub_CMX
    %buf2 = memref.alloc() : !WeightsTableStub_CMX
    %buf3 = memref.alloc() : !WeightsStub_CMX
    %buf4 = memref.alloc() : !WeightsSMStub_CMX

    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<32x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    %0 = VPUIP.Copy inputs(%weight_table : !WeightsTable_DDR) outputs(%buf2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    %1 = VPUIP.Copy inputs(%in : !Input_DDR) outputs(%buf0 : !InputStub_CMX) -> !InputStub_CMX
    %2 = VPUIP.Copy inputs(%weights : !Weights_DDR) outputs(%buf3 : !WeightsStub_CMX) -> !WeightsStub_CMX
    %3 = VPUIP.Copy inputs(%weights_sm : !WeightsSM_DDR) outputs(%buf4 : !WeightsSMStub_CMX) -> !WeightsSMStub_CMX

    %4 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputStub_CMX)
        weights(%2 : !WeightsStub_CMX)
        weights_sparsity_map(%3 : !WeightsSMStub_CMX)
        weight_table(%0 : !WeightsTableStub_CMX)
        parent_input(%1 : !InputStub_CMX)
        parent_output(%buf1 : !OutputStub_CMX)
        outputs(%buf1 : !OutputStub_CMX) -> !OutputStub_CMX
        variants : {
            DPUTask { outEnd = [55, 10, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    return %4 : !OutputStub_CMX


    // CHECK:       [[OUTPUT_MEMREF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[INPUT_MEMREF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[WEIGHT_TABLE_BUFFER:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
    // CHECK:       [[WEIGHTS_BUFFER:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
    // CHECK:       [[WEIGHTS_SM_BUFFER:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>

    // CHECK-DAG:       [[WEIGHT_TABLE:%.+]] = const.Declare memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1> : tensor<32x1x1x4xsi32>, [#const.SwizzleConstant<5 : i64, 3 : i64>]
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>, #const.SwizzleConstant<5 : i64, 3 : i64>]
    // CHECK-DAG:       [[WEIGHTS_SM:%.+]] = const.Declare memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap, #const.SwizzleConstant<5 : i64, 3 : i64>]

    // CHECK:       [[WT:%.*]] = VPUIP.Copy inputs([[WEIGHT_TABLE]] : memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>)
    // CHECK-SAME:                          outputs([[WEIGHT_TABLE_BUFFER]] : memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK-SAME:                          -> memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
    // CHECK:       [[INPUT:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:                             outputs([[OUTPUT_MEMREF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                             -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[W:%.*]] = VPUIP.Copy inputs([[WEIGHTS]] : memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>)
    // CHECK-SAME:                         outputs([[WEIGHTS_BUFFER]] : memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK-SAME:                         -> memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
    // CHECK:       [[W_SM:%.*]] = VPUIP.Copy inputs([[WEIGHTS_SM]] : memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>)
    // CHECK-SAME:                            outputs([[WEIGHTS_SM_BUFFER]] : memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK-SAME:                            -> memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsSMDistributed = !VPUIP.DistributedBuffer<
    32x1x1x128xi1, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x56x56xf16, #NHWC, @DDR>
!Weights_DDR = memref<32x32x1x1xf16, #NHWC, @DDR>
!WeightsSM_DDR = memref<32x1x1x128xi1, @DDR>
!WeightsTable_DDR = memref<32x1x1x4xsi32, @DDR>
!Output_DDR = memref<1x16x56x56xf16, #NHWC, @DDR>

!InputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsSMStub_CMX = memref<32x1x1x128xi1, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @ConvSparseWeightsMulticlustering
func.func @ConvSparseWeightsMulticlustering(%input : !Input_DDR) -> !Output_DDR
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
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
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
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
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

    // Verify that alignment is set for constants which satisfy the 512 size requirement

    // CHECK-DAG:       [[CST_WEIGHT_TABLE:%.+]] = const.Declare memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1> : tensor<32x1x1x4xsi32>, [#const.SwizzleConstant<5 : i64, 3 : i64>]
    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>, #const.SwizzleConstant<5 : i64, 3 : i64>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap, #const.SwizzleConstant<5 : i64, 3 : i64>]

    // CHECK:       [[BUF_INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUF_WEIGHTS_SM:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

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
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!WeightsTable0 = memref<16x1x1x4xsi32, #NHWC, @CMX_NN>
!IODDR0 = memref<1x16x56x56xf16, #NHWC, @DDR>
!IOCMX0 = memref<1x16x56x56xf16, #NHWC, @CMX_NN>
!SMDDR0 = memref<1x16x56x56xi1, #NHWC, @DDR>
!SMCMX0 = memref<1x16x56x56xi1, #NHWC, @CMX_NN>
!WeightsStub = memref<16x16x1x1xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @SetSwizzlingForDpuToDpuBufferInOutSparse
func.func @SetSwizzlingForDpuToDpuBufferInOutSparse(%in_data : !IODDR0,
                        %in_sm : !SMDDR0,
                        %weight_table : !WeightsTable0,
                        %weights : !WeightsStub)
                        -> (!IODDR0, !SMDDR0) {

    %buf0_data = memref.alloc() : !IOCMX0
    %buf0_sm = memref.alloc() : !SMCMX0
    %buf1_data = memref.alloc() : !IOCMX0
    %buf1_sm = memref.alloc() : !SMCMX0

    %buf2_data = memref.alloc() : !IOCMX0
    %buf2_sm = memref.alloc() : !SMCMX0
    %buf3_data = memref.alloc() : !IODDR0
    %buf3_sm = memref.alloc() : !SMDDR0

    %in0_data = VPUIP.Copy
            inputs(%in_data : !IODDR0)
            outputs(%buf0_data : !IOCMX0)
             -> !IOCMX0
    %in0_sm = VPUIP.Copy
            inputs(%in_sm : !SMDDR0)
            outputs(%buf0_sm : !SMCMX0)
             -> !SMCMX0

    %1:2 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%in0_data : !IOCMX0)
        input_sparsity_map(%in0_sm : !SMCMX0)
        weights(%weights : !WeightsStub)
        weight_table(%weight_table : !WeightsTable0)
        parent_input(%in0_data : !IOCMX0)
        parent_input_sparsity_map(%in0_sm : !SMCMX0)
        parent_output(%buf1_data : !IOCMX0)
        parent_output_sparsity_map(%buf1_sm : !SMCMX0)
        outputs(%buf1_data : !IOCMX0)
        output_sparsity_map(%buf1_sm : !SMCMX0)
        -> !IOCMX0, !SMCMX0
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %2:2 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1#0 : !IOCMX0)
        input_sparsity_map(%1#1 : !SMCMX0)
        weights(%weights : !WeightsStub)
        weight_table(%weight_table : !WeightsTable0)
        parent_input(%1#0 : !IOCMX0)
        parent_input_sparsity_map(%1#1 : !SMCMX0)
        parent_output(%buf2_data : !IOCMX0)
        parent_output_sparsity_map(%1#1 : !SMCMX0)
        outputs(%buf2_data : !IOCMX0)
        output_sparsity_map(%buf2_sm : !SMCMX0)
        -> !IOCMX0, !SMCMX0
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %3 = VPUIP.Copy
            inputs(%2#0 : !IOCMX0)
            outputs(%buf3_data : !IODDR0)
             -> !IODDR0
    %4 = VPUIP.Copy
            inputs(%2#1 : !SMCMX0)
            outputs(%buf3_sm : !SMDDR0)
             -> !SMDDR0

    return %3, %4 : !IODDR0, !SMDDR0

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x16x56x56xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_1_DATA:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
    // CHECK:       [[BUFF_1_SM:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<1x16x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
    // CHECK:       [[BUFF_2_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_2_SM:%.+]] = memref.alloc() : memref<1x16x56x56xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_3_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_3_SM:%.+]] = memref.alloc() : memref<1x16x56x56xi1, #NHWC, @DDR>

    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x16x56x56xf16, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs([[BUFF_0_DATA]] : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs(%arg1 : memref<1x16x56x56xi1, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs([[BUFF_0_SM]] : memref<1x16x56x56xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> memref<1x16x56x56xi1, #NHWC, @CMX_NN>

    // CHECK:       [[NCE_0:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[COPY_0]] : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          weights(%arg3 : memref<16x16x1x1xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          parent_input([[COPY_0]] : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          parent_output([[BUFF_1_DATA]] : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>) 
    // CHECK-SAME:          outputs([[BUFF_1_DATA]] : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>) 
      
    // CHECK:       [[NCE_1:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[NCE_0]]#0 : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>) 
    // CHECK-SAME:          weights(%arg3 : memref<16x16x1x1xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          parent_input([[NCE_0]]#0 : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>) 
    // CHECK-SAME:          parent_output([[BUFF_2_DATA]] : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          outputs([[BUFF_2_DATA]] : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) 
      
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[NCE_1]]#0 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:         outputs([[BUFF_3_DATA]] : memref<1x16x56x56xf16, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x16x56x56xf16, #NHWC, @DDR>
    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[NCE_1]]#1 : memref<1x16x56x56xi1, #NHWC, @CMX_NN>) 
    // CHECK-SAME:         outputs([[BUFF_3_SM]] : memref<1x16x56x56xi1, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x16x56x56xi1, #NHWC, @DDR>

    // CHECK:       return [[COPY_2]], [[COPY_3]]
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!WeightsTable0 = memref<16x1x1x4xsi32, #NHWC, @CMX_NN>
!IODDR0 = memref<1x16x56x56xf16, #NHWC, @DDR>
!IOCMX0 = memref<1x16x56x56xf16, #NHWC, @CMX_NN>
!SMDDR0 = memref<1x16x56x56xi1, #NHWC, @DDR>
!SMCMX0 = memref<1x16x56x56xi1, #NHWC, @CMX_NN>
!WeightsStub = memref<16x16x1x1xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @SetSwizzlingForDpuToDpuBufferInSparse
func.func @SetSwizzlingForDpuToDpuBufferInSparse(%in_data : !IODDR0,
                        %in_sm : !SMDDR0,
                        %weight_table : !WeightsTable0,
                        %weights : !WeightsStub)
                        -> !IODDR0 {

    %buf0_data = memref.alloc() : !IOCMX0
    %buf0_sm = memref.alloc() : !SMCMX0
    %buf1_data = memref.alloc() : !IOCMX0

    %buf2_data = memref.alloc() : !IOCMX0
    %buf3_data = memref.alloc() : !IODDR0

    %in0_data = VPUIP.Copy
            inputs(%in_data : !IODDR0)
            outputs(%buf0_data : !IOCMX0)
             -> !IOCMX0
    %in0_sm = VPUIP.Copy
            inputs(%in_sm : !SMDDR0)
            outputs(%buf0_sm : !SMCMX0)
             -> !SMCMX0

    %1 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%in0_data : !IOCMX0)
        input_sparsity_map(%in0_sm : !SMCMX0)
        weights(%weights : !WeightsStub)
        weight_table(%weight_table : !WeightsTable0)
        parent_input(%in0_data : !IOCMX0)
        parent_output(%buf1_data : !IOCMX0)
        outputs(%buf1_data : !IOCMX0)
        -> !IOCMX0
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %2 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !IOCMX0)
        weights(%weights : !WeightsStub)
        weight_table(%weight_table : !WeightsTable0)
        parent_input(%1#0 : !IOCMX0)
        parent_output(%buf2_data : !IOCMX0)
        outputs(%buf2_data : !IOCMX0)
        -> !IOCMX0
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %3 = VPUIP.Copy
            inputs(%2 : !IOCMX0)
            outputs(%buf3_data : !IODDR0)
             -> !IODDR0

    return %3 : !IODDR0

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x16x56x56xi1, #NHWC, @CMX_NN>
    // CHECK:       [[ALLOC_0:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>
}

// ----- 


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// 1x16x170x171xf16 can fit to CMX, but additional in/out sparsity map can not
!WeightsTable0 = memref<16x1x1x4xsi32, #NHWC, @CMX_NN>
!IODDR0 = memref<1x16x170x171xf16, #NHWC, @DDR>
!IOCMX0 = memref<1x16x170x171xf16, #NHWC, @CMX_NN>
!SMDDR0 = memref<1x16x170x171xi1, #NHWC, @DDR>
!SMCMX0 = memref<1x16x170x171xi1, #NHWC, @CMX_NN>
!ActivationWindow0 = memref<16x1x1x16xui8, #NHWC, @CMX_NN>

// CHECK-LABEL: @DoNotSetSwizzlingDueToCmxUsageIncreaseSparse
func.func @DoNotSetSwizzlingDueToCmxUsageIncreaseSparse(%in_data : !IODDR0,
                        %in_sm : !SMDDR0,
                        %weight_table : !WeightsTable0,
                        %act_wind : !ActivationWindow0)
                        -> (!IODDR0, !SMDDR0) {

    %buf0_data = memref.alloc() : !IOCMX0
    %buf0_sm = memref.alloc() : !SMCMX0
    %buf1_data = memref.alloc() : !IOCMX0
    %buf1_sm = memref.alloc() : !SMCMX0

    %buf2_data = memref.alloc() : !IOCMX0
    %buf2_sm = memref.alloc() : !SMCMX0
    %buf3_data = memref.alloc() : !IODDR0
    %buf3_sm = memref.alloc() : !SMDDR0

    %in0_data = VPUIP.Copy
            inputs(%in_data : !IODDR0)
            outputs(%buf0_data : !IOCMX0)
             -> !IOCMX0
    %in0_sm = VPUIP.Copy
            inputs(%in_sm : !SMDDR0)
            outputs(%buf0_sm : !SMCMX0)
             -> !SMCMX0

    %1:2 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%in0_data : !IOCMX0)
        input_sparsity_map(%in0_sm : !SMCMX0)
        weight_table(%weight_table : !WeightsTable0)
        activation_window(%act_wind : !ActivationWindow0)
        parent_input(%in0_data : !IOCMX0)
        parent_input_sparsity_map(%in0_sm : !SMCMX0)
        parent_output(%buf1_data : !IOCMX0)
        parent_output_sparsity_map(%buf1_sm : !SMCMX0)
        outputs(%buf1_data : !IOCMX0)
        output_sparsity_map(%buf1_sm : !SMCMX0)
        -> !IOCMX0, !SMCMX0
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %2:2 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%1#0 : !IOCMX0)
        input_sparsity_map(%1#1 : !SMCMX0)
        weight_table(%weight_table : !WeightsTable0)
        activation_window(%act_wind : !ActivationWindow0)
        parent_input(%1#0 : !IOCMX0)
        parent_input_sparsity_map(%1#1 : !SMCMX0)
        parent_output(%buf2_data : !IOCMX0)
        parent_output_sparsity_map(%1#1 : !SMCMX0)
        outputs(%buf2_data : !IOCMX0)
        output_sparsity_map(%buf2_sm : !SMCMX0)
        -> !IOCMX0, !SMCMX0
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %3 = VPUIP.Copy
            inputs(%2#0 : !IOCMX0)
            outputs(%buf3_data : !IODDR0)
             -> !IODDR0
    %4 = VPUIP.Copy
            inputs(%2#1 : !SMCMX0)
            outputs(%buf3_sm : !SMDDR0)
             -> !SMDDR0

    return %3, %4 : !IODDR0, !SMDDR0

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x16x170x171xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x16x170x171xi1, #NHWC, @CMX_NN>
    // CHECK-NOT:  VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK-NOT:  VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x16x170x171xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_1_SM:%.+]] = memref.alloc() : memref<1x16x170x171xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_2_DATA:%.+]] = memref.alloc() : memref<1x16x170x171xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_2_SM:%.+]] = memref.alloc() : memref<1x16x170x171xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_3_DATA:%.+]] = memref.alloc() : memref<1x16x170x171xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_3_SM:%.+]] = memref.alloc() : memref<1x16x170x171xi1, #NHWC, @DDR>
}

// ----- 

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!WeightsTableDDR = memref<16x1x1x4xsi32, #NHWC, @DDR>
!WeightsTableCMX = memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!DistrWeightsTableCMX = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", 
    num_clusters = 2 : i64
}>

!WeightsDDR = memref<16x16x1x1xf16, #NHWC, @DDR>
!WeightsCMX = memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
!DistrWeightsCMX = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", 
    num_clusters = 2 : i64
}>

!SMCMX0 = memref<1x16x56x56xi1, #NHWC, @CMX_NN>
!SMDDR0 = memref<1x16x56x56xi1, #NHWC, @DDR>
!IODistrCMX0 = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", 
    num_clusters = 2 : i64
}>


!SMCMX1 = memref<1x16x56x56xi1, #NHWC, [@CMX_NN, 0]>
!SMCMX2 = !VPUIP.DistributedBuffer<
    1x16x56x56xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", 
    num_clusters = 2 : i64
}>

!IOCMX0 = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!IOCMX1 = memref<1x16x56x56xf16, #NHWC, @CMX_NN>
!IODDR0 = memref<1x16x56x56xf16, #NHWC, @DDR>

// CHECK-LABEL: @SetSwizzlingForDpuToDpuBufferInMultiClusterSparse
func.func @SetSwizzlingForDpuToDpuBufferInMultiClusterSparse(%arg0: !IODDR0, %arg1: !SMDDR0, %arg2: !WeightsTableDDR, %arg3: !WeightsDDR) -> (!IODDR0, !SMDDR0) {
    %0 = VPURT.AllocDistributed -> !IODistrCMX0
    %1 = VPURT.AllocDistributed -> !SMCMX2
    %2 = VPURT.AllocDistributed -> !DistrWeightsTableCMX
    %3 = VPURT.AllocDistributed -> !DistrWeightsCMX
    %4 = VPURT.AllocDistributed -> !IODistrCMX0
    %5 = VPURT.AllocDistributed -> !SMCMX2
    %6 = VPURT.AllocDistributed -> !IODistrCMX0
    %7 = VPURT.AllocDistributed -> !SMCMX2
    %8 = memref.alloc() : !IODDR0
    %9 = memref.alloc() : !SMDDR0
    %10 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: !IODDR0) outputs(%0 as %arg5: !IOCMX1) -> !IODistrCMX0 {
      %18 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : !IODDR0) outputs(%arg5 : !IOCMX1) -> !IOCMX1
    }
    %11 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: !SMDDR0) outputs(%1 as %arg5: !SMCMX0) -> !SMCMX2 {
      %18 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : !SMDDR0) outputs(%arg5 : !SMCMX0) -> !SMCMX0
    }
    %12 = VPUIP.NCEClusterTiling inputs(%arg2 as %arg4: !WeightsTableDDR) outputs(%2 as %arg5: !WeightsTableCMX) -> !DistrWeightsTableCMX {
      %18 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : !WeightsTableDDR) outputs(%arg5 : !WeightsTableCMX) -> !WeightsTableCMX
    }
    %13 = VPUIP.NCEClusterTiling inputs(%arg3 as %arg4: !WeightsDDR) outputs(%3 as %arg5: !WeightsCMX) -> !DistrWeightsCMX {
      %18 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : !WeightsDDR) outputs(%arg5 : !WeightsCMX) -> !WeightsCMX
    }
    %14:2 = VPUIP.NCEClusterTiling inputs(%10 as %arg4: !IOCMX0, %11 as %arg5: !SMCMX1, %12 as %arg6: !WeightsTableCMX, %13 as %arg7: !WeightsCMX) 
                                   outputs(%4 as %arg8: !IOCMX0, %5 as %arg9: !SMCMX1) -> (!IODistrCMX0, !SMCMX2) {
      %18:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 27 : i64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                                   kernel_strides = [1, 1],
                                   task_type = #VPUIP.nce_task_type<CONV>}
         input(%arg4 : !IOCMX0)
         input_sparsity_map(%arg5 : !SMCMX1)
         weights(%arg7 : !WeightsCMX)
         weight_table(%arg6 : !WeightsTableCMX)
         parent_input(%arg4 : !IOCMX0)
         parent_input_sparsity_map(%arg5 : !SMCMX1)
         parent_output(%arg8 : !IOCMX0)
         parent_output_sparsity_map(%arg9 : !SMCMX1)
         outputs(%arg8 : !IOCMX0)
         output_sparsity_map(%arg9 : !SMCMX1)
         -> !IOCMX0, !SMCMX1 variants : {
        DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [55, 55, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    %15:2 = VPUIP.NCEClusterTiling inputs(%14#0 as %arg4: !IOCMX0, %14#1 as %arg5: !SMCMX1, %12 as %arg6: !WeightsTableCMX, %13 as %arg7: !WeightsCMX) 
                                   outputs(%6 as %arg8: !IOCMX0, %7 as %arg9: !SMCMX1) -> (!IODistrCMX0, !SMCMX2) {
      %18:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 27 : i64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                                   kernel_strides = [1, 1],
                                   task_type = #VPUIP.nce_task_type<CONV>}
         input(%arg4 : !IOCMX0)
         input_sparsity_map(%arg5 : !SMCMX1)
         weights(%arg7 : !WeightsCMX)
         weight_table(%arg6 : !WeightsTableCMX)
         parent_input(%arg4 : !IOCMX0)
         parent_input_sparsity_map(%arg5 : !SMCMX1)
         parent_output(%arg8 : !IOCMX0)
         parent_output_sparsity_map(%arg9 : !SMCMX1)
         outputs(%arg8 : !IOCMX0)
         output_sparsity_map(%arg9 : !SMCMX1)
         -> !IOCMX0, !SMCMX1 variants : {
        DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [55, 55, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    %16 = VPUIP.NCEClusterTiling inputs(%15#0 as %arg4: !IOCMX1) outputs(%8 as %arg5: !IODDR0) -> !IODDR0 {
      %18 = VPUIP.Copy {out_mem_space = @DDR} inputs(%arg4 : !IOCMX1) outputs(%arg5 : !IODDR0) -> !IODDR0
    }
    %17 = VPUIP.NCEClusterTiling inputs(%15#1 as %arg4: !SMCMX0) outputs(%9 as %arg5: !SMDDR0) -> !SMDDR0 {
      %18 = VPUIP.Copy {out_mem_space = @DDR} inputs(%arg4 : !SMCMX0) outputs(%arg5 : !SMDDR0) -> !SMDDR0
    }
    return %16, %17 : !IODDR0, !SMDDR0

    // CHECK:       [[BUFF_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    
    // CHECK:       [[BUFF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_W:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[BUFF_2_DATA:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_2_SM:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x16x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_3_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_3_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_4_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_4_SM:%.+]] = memref.alloc() : memref<1x16x56x56xi1, #NHWC, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: memref<1x16x56x56xf16, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs([[BUFF_0_DATA]] as %arg5: memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:       [[inner_0:%.+]] = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : memref<1x16x56x56xf16, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs(%arg5 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>
      
    // CHECK:       [[COPY_1:%.+]] = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: memref<1x16x56x56xi1, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs([[BUFF_0_SM]] as %arg5: memref<1x16x56x56xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x16x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:       [[inner_1:%.+]] = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : memref<1x16x56x56xi1, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs(%arg5 : memref<1x16x56x56xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> memref<1x16x56x56xi1, #NHWC, @CMX_NN>
      
    // CHECK:       [[COPY_2:%.+]] = VPUIP.NCEClusterTiling inputs(%arg2 as %arg4: memref<16x1x1x4xsi32, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs([[BUFF_1]] as %arg5: memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:       [[inner_2:%.+]] = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : memref<16x1x1x4xsi32, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs(%arg5 : memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          -> memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
      
    // CHECK:       [[COPY_3:%.+]] = VPUIP.NCEClusterTiling inputs(%arg3 as %arg4: memref<16x16x1x1xf16, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs([[BUFF_W]] as %arg5: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:       [[inner_3:%.+]] = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : memref<16x16x1x1xf16, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs(%arg5 : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
      
    // CHECK:       [[NCE_0:%.+]]:2 = VPUIP.NCEClusterTiling inputs([[COPY_0]] as %arg4: memref<1x16x56x56xf16, #NHWC, @CMX_NN>, [[COPY_1]] as %arg5: memref<1x16x56x56xi1, #NHWC, @CMX_NN>, [[COPY_2]] as %arg6: memref<16x1x1x4xsi32, #NHWC, @CMX_NN>, [[COPY_3]] as %arg7: memref<16x16x1x1xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:         outputs([[BUFF_2_DATA]] as %arg8: memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>, [[BUFF_2_SM]] as %arg9: memref<1x16x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK-SAME:          -> (!VPUIP.DistributedBuffer<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x16x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) {
    // CHECK:       [[inner_4:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input(%arg4 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          weights(%arg7 : memref<16x16x1x1xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          parent_input(%arg4 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:          parent_output(%arg8 : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>) 
    // CHECK-SAME:          outputs(%arg8 : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>) 
    // CHECK-SAME:          output_sparsity_map(%arg9 : memref<1x16x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)

    // CHECK:       [[NCE_1:%.+]]:2 = VPUIP.NCEClusterTiling inputs([[NCE_0]]#0 as %arg4: memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>, [[NCE_0]]#1 as %arg5: memref<1x16x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>, [[COPY_2]] as %arg6: memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>, [[COPY_3]] as %arg7: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) 
    // CHECK-SAME:         outputs([[BUFF_3_DATA]] as %arg8: memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>, [[BUFF_3_SM]] as %arg9: memref<1x16x56x56xi1, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          -> (!VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x16x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) {
    // CHECK:       [[inner_5:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input(%arg4 : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>) 
    // CHECK-SAME:          input_sparsity_map(%arg5 : memref<1x16x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK-SAME:          weights(%arg7 : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) 
    // CHECK-SAME:          parent_input(%arg4 : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>) 
    // CHECK-SAME:          parent_output(%arg8 : memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>) 
    // CHECK-SAME:          outputs(%arg8 : memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>) 
      
    // CHECK:       [[COPY_4:%.+]] = VPUIP.NCEClusterTiling inputs([[NCE_1]]#0 as %arg4: memref<1x16x56x56xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:         outputs([[BUFF_4_DATA]] as %arg5: memref<1x16x56x56xf16, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x16x56x56xf16, #NHWC, @DDR> {
    // CHECK:       [[inner_6:%.+]] = VPUIP.Copy {out_mem_space = @DDR} inputs(%arg4 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) 
    // CHECK-SAME:         outputs(%arg5 : memref<1x16x56x56xf16, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x16x56x56xf16, #NHWC, @DDR>
      
    // CHECK:       [[COPY_5:%.+]] = VPUIP.NCEClusterTiling inputs([[NCE_1]]#1 as %arg4: memref<1x16x56x56xi1, #NHWC, @CMX_NN>) 
    // CHECK-SAME:         outputs([[BUFF_4_SM]] as %arg5: memref<1x16x56x56xi1, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x16x56x56xi1, #NHWC, @DDR> {
    // CHECK:       [[inner_7:%.+]] = VPUIP.Copy {out_mem_space = @DDR} inputs(%arg4 : memref<1x16x56x56xi1, #NHWC, @CMX_NN>) 
    // CHECK-SAME:         outputs(%arg5 : memref<1x16x56x56xi1, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x16x56x56xi1, #NHWC, @DDR>
      
    // CHECK:       return [[COPY_4]], [[COPY_5]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!WeightsSm0 = memref<32x1x1x128xi1, @DDR>
!IOCMX0 = memref<1x32x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsTable0 = memref<32x1x1x4xsi32, #NHWC, @DDR>
!ActivationWindow0 = memref<32x1x1x16xui8, #NHWC, @DDR>
!SMCMX0 = !VPUIP.DistributedBuffer<
    1x32x56x56xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", 
    num_clusters = 2 : i64
}>

!IODistrCMX0 = !VPUIP.DistributedBuffer<
    1x32x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", 
    num_clusters = 2 : i64
}>

!Weights1 = !VPUIP.DistributedBuffer<
    32x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", 
    num_clusters = 2 : i64
}>

!WeightsTable1 = memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!ActivationWindow1 = !VPUIP.DistributedBuffer<
    32x1x1x16xui8, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", 
    num_clusters = 2 : i64
}>

!Weights2 = memref<32x1x1x128xi1, @CMX_NN>
!SMCMX1 = memref<1x32x56x56xi1, #NHWC, @CMX_NN>
!ActivationWindow2 = memref<32x1x1x16xui8, #NHWC, [@CMX_NN, 0]>
!Weights3 = !VPUIP.DistributedBuffer<
    32x1x1x128xi1, #NCHW, 
    @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
}>

!Weights4 = memref<32x32x1x1xf16, #NHWC, @DDR>
!IOCMX1 = memref<1x32x56x56xf16, #NHWC, @CMX_NN>
!WeightsTable2 = memref<32x1x1x4xsi32, @DDR>
!WeightsTable3 = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", 
    num_clusters = 2 : i64
}>

!WeightsTable4 = memref<32x1x1x4xsi32, [@CMX_NN, 0]>
!IODDR0 = memref<1x32x56x56xf16, #NHWC, @DDR>
!SMDDR0 = memref<1x32x56x56xi1, #NHWC, @DDR>
!Weights5 = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!SMCMX2 = memref<1x32x56x56xi1, #NHWC, [@CMX_NN, 0]>

// [Track number: E#65141]

// CHECK-LABEL: @SetSwizzlingForDpuToDpuBufferWithWeightsInMultiClusterSparse
func.func @SetSwizzlingForDpuToDpuBufferWithWeightsInMultiClusterSparse(%arg0: !IODDR0, %arg1: !SMDDR0) -> (!IODDR0, !SMDDR0) {
    %cst_aw0 = const.Declare !ActivationWindow0 = dense<1> : tensor<32x1x1x16xui8>, [#const.Reorder<#NHWC>]
    %cst_aw2 = const.Declare !ActivationWindow0 = dense<1> : tensor<32x1x1x16xui8>, [#const.Reorder<#NHWC>]
    %cst_wt0 = const.Declare !WeightsTable0 = dense<1> : tensor<32x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %cst_wt2 = const.Declare !WeightsTable2 = dense<1> : tensor<32x1x1x4xsi32>
    %cst_0 = const.Declare !Weights4 = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_1 = const.Declare !WeightsSm0 = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    %buf0_data = VPURT.AllocDistributed -> !IODistrCMX0
    %buf0_sm = VPURT.AllocDistributed -> !SMCMX0

    %buf1_data = VPURT.AllocDistributed -> !IODistrCMX0
    %buf1_sm = VPURT.AllocDistributed -> !SMCMX0

    %buf2_data = VPURT.AllocDistributed -> !IODistrCMX0
    %buf2_sm = VPURT.AllocDistributed -> !SMCMX0

    %convW = VPURT.AllocDistributed -> !Weights1
    %convWSM = VPURT.AllocDistributed -> !Weights3
    %convWT = VPURT.AllocDistributed -> !WeightsTable3

    %maxpoolWT = VPURT.AllocDistributed -> !WeightsTable3
    %actwindow = VPURT.AllocDistributed -> !ActivationWindow1
    %actwindow2 = VPURT.AllocDistributed -> !ActivationWindow1

    %outData = memref.alloc() : !IODDR0
    %outSM  = memref.alloc() : !SMDDR0

    %13 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: !IODDR0) outputs(%buf0_data as %arg5: !IOCMX1) -> !IODistrCMX0 {
      %24 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : !IODDR0) outputs(%arg5 : !IOCMX1) -> !IOCMX1
    }
    %14 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: !SMDDR0) outputs(%buf0_sm as %arg5: !SMCMX1) -> !SMCMX0 {
      %24 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : !SMDDR0) outputs(%arg5 : !SMCMX1) -> !SMCMX1
    }
    %15 = VPUIP.NCEClusterTiling inputs(%cst_wt0 as %arg4: !WeightsTable0) outputs(%maxpoolWT as %arg5: !WeightsTable1) -> !WeightsTable3 {
      %24 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : !WeightsTable0) outputs(%arg5 : !WeightsTable1) -> !WeightsTable1
    }
    %16 = VPUIP.NCEClusterTiling inputs(%cst_aw0 as %arg4: !ActivationWindow0) outputs(%actwindow as %arg5: !ActivationWindow2) -> !ActivationWindow1 {
      %24 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : !ActivationWindow0) outputs(%arg5 : !ActivationWindow2) -> !ActivationWindow2
    }
    %17 = VPUIP.NCEClusterTiling inputs(%cst_0 as %arg4: !Weights4) outputs(%convW as %arg5: !Weights5) -> !Weights1 {
      %24 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : !Weights4) outputs(%arg5 : !Weights5) -> !Weights5
    }
    %18 = VPUIP.NCEClusterTiling inputs(%cst_1 as %arg4: !WeightsSm0) outputs(%convWSM as %arg5: !Weights2) -> !Weights3 {
      %24 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : !WeightsSm0) outputs(%arg5 : !Weights2) -> !Weights2
    }
    %19 = VPUIP.NCEClusterTiling inputs(%cst_wt2 as %arg4: !WeightsTable2) outputs(%convWT as %arg5: !WeightsTable4) -> !WeightsTable3 {
      %24 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : !WeightsTable2) outputs(%arg5 : !WeightsTable4) -> !WeightsTable4
    }
    %25 = VPUIP.NCEClusterTiling inputs(%cst_aw2 as %arg4: !ActivationWindow0) outputs(%actwindow2 as %arg5: !ActivationWindow2) -> !ActivationWindow1 {
      %24 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg4 : !ActivationWindow0) outputs(%arg5 : !ActivationWindow2) -> !ActivationWindow2
    }
    %20:2 = VPUIP.NCEClusterTiling inputs(%13 as %arg4: !IOCMX0, %14 as %arg5: !SMCMX2, %19 as %arg6: !WeightsTable4, %16 as %arg7: !ActivationWindow2, %17 as %arg8: !Weights5, %18 as %arg9: !Weights2) 
                                   outputs(%buf1_data as %arg10: !IOCMX0, %buf1_sm as %arg11: !SMCMX2) -> (!IODistrCMX0, !SMCMX0) {
      %24:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 27 : i64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                                   kernel_strides = [1, 1],
                                   task_type = #VPUIP.nce_task_type<CONV>}
         input(%arg4 : !IOCMX0)
         input_sparsity_map(%arg5 : !SMCMX2)
         weights(%arg8 : !Weights5)
         weights_sparsity_map(%arg9 : !Weights2)
         weight_table(%arg6 : !WeightsTable4)
         activation_window(%arg7 : !ActivationWindow2)
         parent_input(%arg4 : !IOCMX0)
         parent_input_sparsity_map(%arg5 : !SMCMX2)
         parent_output(%arg10 : !IOCMX0)
         parent_output_sparsity_map(%arg11 : !SMCMX2)
         outputs(%arg10 : !IOCMX0)
         output_sparsity_map(%arg11 : !SMCMX2)
         -> !IOCMX0, !SMCMX2 variants : {
        DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [55, 55, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    %21:2 = VPUIP.NCEClusterTiling inputs(%20#0 as %arg4: !IOCMX0, %20#1 as %arg5: !SMCMX2, %15 as %arg6: !WeightsTable1, %25 as %arg7: !ActivationWindow2) 
                                   outputs(%buf2_data as %arg8: !IOCMX0, %buf2_sm as %arg9: !SMCMX2) -> (!IODistrCMX0, !SMCMX0) {
      %24:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 27 : i64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                                   kernel_strides = [1, 1],
                                   task_type = #VPUIP.nce_task_type<MAXPOOL>}
         input(%arg4 : !IOCMX0)
         input_sparsity_map(%arg5 : !SMCMX2)
         weight_table(%arg6 : !WeightsTable1)
         activation_window(%arg7 : !ActivationWindow2)
         parent_input(%arg4 : !IOCMX0)
         parent_input_sparsity_map(%arg5 : !SMCMX2)
         parent_output(%arg8 : !IOCMX0)
         parent_output_sparsity_map(%arg9 : !SMCMX2)
         outputs(%arg8 : !IOCMX0)
         output_sparsity_map(%arg9 : !SMCMX2)
         -> !IOCMX0, !SMCMX2 variants : {
        DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [55, 55, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      }
    }
    %22 = VPUIP.NCEClusterTiling inputs(%21#0 as %arg4: !IOCMX1) outputs(%outData as %arg5: !IODDR0) -> !IODDR0 {
      %24 = VPUIP.Copy {out_mem_space = @DDR} inputs(%arg4 : !IOCMX1) outputs(%arg5 : !IODDR0) -> !IODDR0
    }
    %23 = VPUIP.NCEClusterTiling inputs(%21#1 as %arg4: !SMCMX1) outputs(%outSM  as %arg5: !SMDDR0) -> !SMDDR0 {
      %24 = VPUIP.Copy {out_mem_space = @DDR} inputs(%arg4 : !SMCMX1) outputs(%arg5 : !SMDDR0) -> !SMDDR0
    }
    return %22, %23 : !IODDR0, !SMDDR0

    // CHECK-DAG:       [[CST_AW0:%.+]] = const.Declare memref<32x1x1x16xui8, #NHWC, @DDR> = dense<1> : tensor<32x1x1x16xui8>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_AW1:%.+]] = const.Declare memref<32x1x1x16xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1> : tensor<32x1x1x16xui8>, [#const.Reorder<#NHWC>, #const.SwizzleConstant<5 : i64, 3 : i64>]
    // CHECK-DAG:       [[CST_WT1:%.+]] = const.Declare memref<32x1x1x4xsi32, #NHWC, @DDR> = dense<1> : tensor<32x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_WT0:%.+]] = const.Declare memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1> : tensor<32x1x1x4xsi32>, [#const.SwizzleConstant<5 : i64, 3 : i64>]
    // CHECK-DAG:       [[CST_W:%.+]] = const.Declare memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>, #const.SwizzleConstant<5 : i64, 3 : i64>]
    // CHECK-DAG:       [[CST_W_SM:%.+]] = const.Declare memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap, #const.SwizzleConstant<5 : i64, 3 : i64>]

    // CHECK:       [[BUFF_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_1_DATA:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x32x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_1_SM:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x32x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_2_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_2_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_3_DATA:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_3_SM:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_4:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_5:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_6:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x1x1x16xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_7:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x16xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_8_DATA:%.+]] = memref.alloc() : memref<1x32x56x56xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_8_SM:%.+]] = memref.alloc() : memref<1x32x56x56xi1, #NHWC, @DDR>

    // CHECK:       [[COPY_0_ACT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg2: memref<1x32x56x56xf16, #NHWC, @DDR>) 
    // CHECK-SAME:      outputs([[BUFF_0_DATA]] as %arg3: memref<1x32x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK:           VPUIP.Copy
    // CHECK-SAME:         inputs(%arg2 : memref<1x32x56x56xf16, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs(%arg3 : memref<1x32x56x56xf16, #NHWC, @CMX_NN>)
      
    // CHECK:       [[COPY_1_SM:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:         inputs(%arg1 as %arg2: memref<1x32x56x56xi1, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs([[BUFF_0_SM]] as %arg3: memref<1x32x56x56xi1, #NHWC, @CMX_NN>)
    // CHECK:              VPUIP.Copy
    // CHECK-SAME:           inputs(%arg2 : memref<1x32x56x56xi1, #NHWC, @DDR>) 
    // CHECK-SAME:           outputs(%arg3 : memref<1x32x56x56xi1, #NHWC, @CMX_NN>)

    // CHECK:       [[COPY_WT1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:         inputs([[CST_WT1]] as %arg2: memref<32x1x1x4xsi32, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs([[BUFF_5]] as %arg3: memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>)
    // CHECK:              VPUIP.Copy
    // CHECK-SAME:           inputs(%arg2 : memref<32x1x1x4xsi32, #NHWC, @DDR>) 
    // CHECK-SAME:           outputs(%arg3 : memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       [[COPY_AW0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:         inputs([[CST_AW0]] as %arg2: memref<32x1x1x16xui8, #NHWC, @DDR>) 
    // CHECK-SAME:         outputs([[BUFF_6]] as %arg3: memref<32x1x1x16xui8, #NHWC, [@CMX_NN, 0]>)
    // CHECK:              VPUIP.Copy
    // CHECK-SAME:           inputs(%arg2 : memref<32x1x1x16xui8, #NHWC, @DDR>) 
    // CHECK-SAME:           outputs(%arg3 : memref<32x1x1x16xui8, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       [[COPY_W:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:         inputs([[CST_W]] as %arg2: memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>) 
    // CHECK-SAME:         outputs([[BUFF_3_DATA]] as %arg3: memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK:              VPUIP.Copy
    // CHECK-SAME:           inputs(%arg2 : memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>) 
    // CHECK-SAME:           outputs(%arg3 : memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)

    // CHECK:       [[COPY_W_SM:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:         inputs([[CST_W_SM]] as %arg2: memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>) 
    // CHECK-SAME:         outputs([[BUFF_3_SM]] as %arg3: memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK:              VPUIP.Copy
    // CHECK-SAME:           inputs(%arg2 : memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>) 
    // CHECK-SAME:           outputs(%arg3 : memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)

    // CHECK:       [[COPY_WT0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:         inputs([[CST_WT0]] as %arg2: memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>) 
    // CHECK-SAME:         outputs([[BUFF_4]] as %arg3: memref<32x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK:              VPUIP.Copy
    // CHECK-SAME:           inputs(%arg2 : memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>) 
    // CHECK-SAME:           outputs(%arg3 : memref<32x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)

    // CHECK:       [[COPY_AW1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:         inputs([[CST_AW1]] as %arg2: memref<32x1x1x16xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>) 
    // CHECK-SAME:         outputs([[BUFF_7]] as %arg3: memref<32x1x1x16xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK:              VPUIP.Copy
    // CHECK-SAME:           inputs(%arg2 : memref<32x1x1x16xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>) 
    // CHECK-SAME:           outputs(%arg3 : memref<32x1x1x16xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)

    // CHECK:       [[NCE_0:%.+]]:2 = VPUIP.NCEClusterTiling 
    // CHECK-SAME:         inputs([[COPY_0_ACT]] as %arg2: memref<1x32x56x56xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                [[COPY_1_SM]] as %arg3: memref<1x32x56x56xi1, #NHWC, @CMX_NN>,
    // CHECK-SAME:                [[COPY_WT0]] as %arg4: memref<32x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>,
    // CHECK-SAME:                [[COPY_AW0]] as %arg5: memref<32x1x1x16xui8, #NHWC, @CMX_NN>,
    // CHECK-SAME:                [[COPY_W]] as %arg6: memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>,
    // CHECK-SAME:                [[COPY_W_SM]] as %arg7: memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>) 
    // CHECK-SAME:         outputs(
    // CHECK-SAME:             [[BUFF_1_DATA]] as %arg8: memref<1x32x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
    // CHECK-SAME:             [[BUFF_1_SM]] as %arg9: memref<1x32x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:          input(%arg2 : memref<1x32x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          input_sparsity_map(%arg3 : memref<1x32x56x56xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:          weights(%arg6 : memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>) 
    // CHECK-SAME:          weights_sparsity_map(%arg7 : memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}
    // CHECK-SAME:          weight_table(%arg4 : memref<32x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>) 
    // CHECK-SAME:          activation_window(%arg5 : memref<32x1x1x16xui8, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg8 : memref<1x32x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK-SAME:          output_sparsity_map(%arg9 : memref<1x32x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
      
    // CHECK:       [[NCE_1:%.+]]:2 = VPUIP.NCEClusterTiling 
    // CHECK-SAME:         inputs([[NCE_0]]#0 as %arg2: memref<1x32x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>,
    // CHECK-SAME:                [[NCE_0]]#1 as %arg3: memref<1x32x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>,
    // CHECK-SAME:                [[COPY_WT1]] as %arg4: memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>,
    // CHECK-SAME:                [[COPY_AW1]] as %arg5: memref<32x1x1x16xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK-SAME:         outputs(
    // CHECK-SAME:             [[BUFF_2_DATA]] as %arg6: memref<1x32x56x56xf16, #NHWC, [@CMX_NN, 0]>,
    // CHECK-SAME:             [[BUFF_2_SM]] as %arg7: memref<1x32x56x56xi1, #NHWC, [@CMX_NN, 0]>)
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:          input(%arg2 : memref<1x32x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK-SAME:          input_sparsity_map(%arg3 : memref<1x32x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK-SAME:          weight_table(%arg4 : memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          activation_window(%arg5 : memref<32x1x1x16xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg6 : memref<1x32x56x56xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          output_sparsity_map(%arg7 : memref<1x32x56x56xi1, #NHWC, [@CMX_NN, 0]>)        
}

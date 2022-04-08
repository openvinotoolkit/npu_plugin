//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-vpu-nce-to-vpuip --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Conv
func @Conv(%arg0: memref<1x16x16x16xf16, #NHWC, @CMX_NN>, %arg1: memref<16x16x1x1xf16, #NHWC, @CMX_NN>, %arg2 : memref<16x1x1x4xsi32, @CMX_NN>)
        -> memref<1x16x16x16xf16, #NHWC, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
        to tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<16x16x1x1xf16, #NHWC, @CMX_NN>
        to tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = builtin.unrealized_conversion_cast %arg2 : memref<16x1x1x4xsi32, @CMX_NN>
        to tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>

    %3 = VPU.NCE.Convolution(%0, %1, %2) {
                pad = {bottom = 0, left = 0, right = 0, top = 0},
                rawFilterShape = [16, 16, 1, 1],
                ppe = {clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = "LRELU"},
                strides = [1, 1]
            } -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload [0, 0, 0, 0] [1, 16, 3, 16] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        VPU.DPU.Workload [0, 0, 3, 0] [1, 16, 3, 16] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        VPU.DPU.Workload [0, 0, 6, 0] [1, 16, 3, 16] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        VPU.DPU.Workload [0, 0, 9, 0] [1, 16, 3, 16] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        VPU.DPU.Workload [0, 0, 12, 0] [1, 16, 4, 16] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
    }

    %4 = builtin.unrealized_conversion_cast %3 : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        to memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    return %4 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:          kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = "CONV"
    // CHECK-SAME:      input(%arg0 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights(%arg1 : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table(%arg2 : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input(%arg0 : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:           DPUTask
    // CHECK-SAME:          mpe_mode = "VECTOR_FP16",
    // CHECK-SAME:          outEnd = [15, 2, 15],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK:           DPUTask
    // CHECK-SAME:          mpe_mode = "VECTOR_FP16",
    // CHECK-SAME:          outEnd = [15, 5, 15],
    // CHECK-SAME:          outStart = [0, 3, 0],
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK:           DPUTask
    // CHECK-SAME:          mpe_mode = "VECTOR_FP16",
    // CHECK-SAME:          outEnd = [15, 8, 15],
    // CHECK-SAME:          outStart = [0, 6, 0],
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK:           DPUTask
    // CHECK-SAME:          mpe_mode = "VECTOR_FP16",
    // CHECK-SAME:          outEnd = [15, 11, 15],
    // CHECK-SAME:          outStart = [0, 9, 0],
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK:           DPUTask
    // CHECK-SAME:          mpe_mode = "VECTOR_FP16",
    // CHECK-SAME:          outEnd = [15, 15, 15],
    // CHECK-SAME:          outStart = [0, 12, 0],
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK:           PPETask "LRELU" {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!SparseInputActBufferType = type !VPUIP.SparseBuffer<
    data=memref<1x16x224x224xf16, #NHWC, @CMX_NN>, 
    sparsity_map=memref<1x16x224x224xi1, #NHWC, @CMX_NN>
>
!SparseInputActTensorType = type !VPU.SparseTensor<
    data=tensor<1x16x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>, 
    sparsity_map=tensor<1x16x224x224xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

!WeightsBufferType = type memref<32x16x7x7xf16, #NHWC, @CMX_NN>
!WeightsTensorType = type tensor<32x16x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>

!WeightsTableTensorType = type tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!WeightsTableBufType = type memref<32x1x1x4xsi32, @CMX_NN>

!SparseConvOutputTensorType = type !VPU.SparseTensor<
    data=tensor<1x32x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    sparsity_map=tensor<1x32x112x112xi1, {mem_space = @CMX_NN, order = #NHWC}>
>

!SparseConvOutputBufferType = type !VPUIP.SparseBuffer<
    data=memref<1x32x112x112xf16, #NHWC, @CMX_NN>, 
    sparsity_map=memref<1x32x112x112xi1, #NHWC, @CMX_NN>
>

// CHECK-LABEL: @SparseActConv
func @SparseActConv(%arg0:  !SparseInputActBufferType,
                    %arg1 : !WeightsBufferType,
                    %arg2 : !WeightsTableBufType) -> !SparseConvOutputBufferType {

    %0 = builtin.unrealized_conversion_cast %arg0 : !SparseInputActBufferType to !SparseInputActTensorType
    %1 = builtin.unrealized_conversion_cast %arg1 : !WeightsBufferType to !WeightsTensorType
    %2 = builtin.unrealized_conversion_cast %arg2 : !WeightsTableBufType to !WeightsTableTensorType

    %3 = VPU.NCE.Convolution(%0, %1, %2) {
                pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64},
                ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELU"},
                rawFilterShape = [32, 16, 7, 7], strides = [2, 2], 
                tilingStrategy = [1, 2, 1, 1]
            } -> !SparseConvOutputTensorType {
        VPU.DPU.Workload [0, 0, 0, 0] [1, 32, 56, 112] {bottom = 0 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64} "CUBOID_16x16" attributes {cluster_id = 0 : i64}
        VPU.DPU.Workload [0, 0, 56, 0] [1, 32, 56, 112] {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 0 : i64} "CUBOID_16x16" attributes {cluster_id = 1 : i64}
    }
    
    %4 = builtin.unrealized_conversion_cast %3 : !SparseConvOutputTensorType to !SparseConvOutputBufferType

    return %4 : !SparseConvOutputBufferType

    //CHECK:        [[OUT_BUF:%.+]] = memref.alloc() : memref<1x32x112x112xf16, #NHWC, @CMX_NN>
    //CHECK:        [[OUT_SM_BUF:%.+]] = memref.alloc() : memref<1x32x112x112xi1, #NHWC, @CMX_NN>
    
    //CHECK:        [[IN_BUF:%.+]], [[IN_SM:%.+]] = VPUIP.UngroupSparseBuffer(%arg0) 
    //CHECK-SAME:                         {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} 
    //CHECK-SAME:                         -> memref<1x16x224x224xf16, #NHWC, @CMX_NN>, memref<1x16x224x224xi1, #NHWC, @CMX_NN>
    
    //CHECK:        [[OUT_DATA_SM:%.+]]:2 = VPUIP.NCEClusterTask
    //CHECK-SAME:               kernel_padding = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}
    //CHECK-SAME:               kernel_size = [7, 7],
    //CHECK-SAME:               kernel_strides = [2, 2],
    //CHECK-SAME:               task_type = "CONV"
    //CHECK-SAME:           input([[IN_BUF]] : memref<1x16x224x224xf16, #NHWC, @CMX_NN>) 
    //CHECK-SAME:           input_sparsity_map([[IN_SM]] : memref<1x16x224x224xi1, #NHWC, @CMX_NN>) 
    //CHECK-SAME:           weights(%arg1 : memref<32x16x7x7xf16, #NHWC, @CMX_NN>) 
    //CHECK-SAME:           weight_table(%arg2 : memref<32x1x1x4xsi32, @CMX_NN>) 
    //CHECK-SAME:           parent_input([[IN_BUF]] : memref<1x16x224x224xf16, #NHWC, @CMX_NN>) 
    //CHECK-SAME:           parent_input_sparsity_map([[IN_SM]] : memref<1x16x224x224xi1, #NHWC, @CMX_NN>) 
    //CHECK-SAME:           parent_output([[OUT_BUF]] : memref<1x32x112x112xf16, #NHWC, @CMX_NN>) 
    //CHECK-SAME:           parent_output_sparsity_map([[OUT_SM_BUF]] : memref<1x32x112x112xi1, #NHWC, @CMX_NN>) 
    //CHECK-SAME:           outputs([[OUT_BUF]] : memref<1x32x112x112xf16, #NHWC, @CMX_NN>) 
    //CHECK-SAME:           output_sparsity_map([[OUT_SM_BUF]] : memref<1x32x112x112xi1, #NHWC, @CMX_NN>) 
    //CHECK-SAME:               -> memref<1x32x112x112xf16, #NHWC, @CMX_NN>, memref<1x32x112x112xi1, #NHWC, @CMX_NN> variants : {
    
    //CHECK:        DPUTask
    //CHECK-SAME:    cluster_id = 0 : i64, 
    //CHECK-SAME:    mpe_mode = "CUBOID_16x16", 
    //CHECK-SAME:    outEnd = [111, 55, 31], 
    //CHECK-SAME:    outStart = [0, 0, 0], 
    //CHECK-SAME:    pad = {bottom = 0 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}}
    
    //CHECK:        DPUTask 
    //CHECK-SAME:    cluster_id = 1 : i64, 
    //CHECK-SAME:    mpe_mode = "CUBOID_16x16", 
    //CHECK-SAME:    outEnd = [111, 111, 31], 
    //CHECK-SAME:    outStart = [0, 56, 0], 
    //CHECK-SAME:    pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 0 : i64}}

    //CHECK:        PPETask "LRELU"
    //CHECK-SAME:    clamp_high = 2147483647 : i64, 
    //CHECK-SAME:    clamp_low = -2147483648 : i64, 
    //CHECK-SAME:    fp_prelu_alpha = 1.000000e+00 : f64, 
    //CHECK-SAME:    lrelu_mult = 1 : i64, 
    //CHECK-SAME:    lrelu_shift = 0 : i64


    //CHECK:        [[OUT_GROUPED_BUF:%.+]] = VPUIP.GroupSparseBuffer([[OUT_DATA_SM]]#0, [[OUT_DATA_SM]]#1)
    //CHECK-SAME:   -> !VPUIP.SparseBuffer<data=memref<1x32x112x112xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x112x112xi1, #NHWC, @CMX_NN>>

    //CHECK:        return [[OUT_GROUPED_BUF]] : !VPUIP.SparseBuffer<data=memref<1x32x112x112xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x112x112xi1, #NHWC, @CMX_NN>>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputActBufferType = type memref<1x64x56x56xf16, #NHWC, @CMX_NN>
!InputActTensorType = type tensor<1x64x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>

!SparseWeightsBufferType = type !VPUIP.SparseBuffer<data=memref<64x64x1x1xf16, #NHWC, @CMX_NN>, sparsity_map=memref<64x1x1x128xi1, @CMX_NN>, is_weights>
!SparseWeightsTensorType = type !VPU.SparseTensor<data=tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<64x1x1x128xi1, {mem_space = @CMX_NN, order = #NCHW}>, is_weights>

!WeightsTableTensorType = type tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!WeightsTableBufType = type memref<64x1x1x4xsi32, @CMX_NN>

!ConvOutputTensorType = type tensor<1x64x56x56xf16, {mem_space = @CMX_NN, order = #NHWC}>
!ConvOutputBufferType = type memref<1x64x56x56xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @SparseWeightsConv
func @SparseWeightsConv(%arg0: !InputActBufferType,
                        %arg1 : !SparseWeightsBufferType,
                        %arg2 : !WeightsTableBufType) -> !ConvOutputBufferType {

    %0 = builtin.unrealized_conversion_cast %arg0 : !InputActBufferType to !InputActTensorType
    %1 = builtin.unrealized_conversion_cast %arg1 : !SparseWeightsBufferType to !SparseWeightsTensorType
    %2 = builtin.unrealized_conversion_cast %arg2 : !WeightsTableBufType to !WeightsTableTensorType

    %3 = VPU.NCE.Convolution(%0, %1, %2) {
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELU"},
                rawFilterShape = [64, 64, 1, 1], strides = [1, 1]
            } -> !ConvOutputTensorType {
        VPU.DPU.Workload [0, 0, 0, 0] [1, 64, 28, 56] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_16x16" attributes {cluster_id = 0 : i64}
        VPU.DPU.Workload [0, 0, 28, 0] [1, 64, 28, 56] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_16x16" attributes {cluster_id = 1 : i64}
    }

    %4 = builtin.unrealized_conversion_cast %3 : !ConvOutputTensorType to !ConvOutputBufferType

    return %4 : !ConvOutputBufferType

//CHECK:        [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x56x56xf16, #NHWC, @CMX_NN>

//CHECK:        [[WEIGHTS_DATA:%.+]], [[WEIGHTS_SM:%.+]] = VPUIP.UngroupSparseBuffer(%arg1)
//CHECK-SAME:                                           {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
//CHECK-SAME:                                           -> memref<64x64x1x1xf16, #NHWC, @CMX_NN>, memref<64x1x1x128xi1, @CMX_NN>

//CHECK:        [[CONV_OUTPUT:%.+]] = VPUIP.NCEClusterTask
//CHECK-SAME:                           kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
//CHECK-SAME:                           kernel_size = [1, 1],
//CHECK-SAME:                           kernel_strides = [1, 1],
//CHECK-SAME:                           task_type = "CONV"
//CHECK-SAME:               input(%arg0 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
//CHECK-SAME:               weights([[WEIGHTS_DATA]] : memref<64x64x1x1xf16, #NHWC, @CMX_NN>)
//CHECK-SAME:               weights_sparsity_map([[WEIGHTS_SM]] : memref<64x1x1x128xi1, @CMX_NN>)
//CHECK-SAME:               weight_table(%arg2 : memref<64x1x1x4xsi32, @CMX_NN>)
//CHECK-SAME:               parent_input(%arg0 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
//CHECK-SAME:               parent_output([[OUT_BUF]] : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
//CHECK-SAME:               outputs([[OUT_BUF]] : memref<1x64x56x56xf16, #NHWC, @CMX_NN>) 
//CHECK-SAME:                   -> memref<1x64x56x56xf16, #NHWC, @CMX_NN> variants : {

//CHECK:        DPUTask
//CHECK-SAME:       cluster_id = 0 : i64,
//CHECK-SAME:       mpe_mode = "CUBOID_16x16",
//CHECK-SAME:       outEnd = [55, 27, 63],
//CHECK-SAME:       outStart = [0, 0, 0],
//CHECK-SAME:       pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}

//CHECK:        DPUTask
//CHECK-SAME:       cluster_id = 1 : i64,
//CHECK-SAME:       mpe_mode = "CUBOID_16x16",
//CHECK-SAME:       outEnd = [55, 55, 63],
//CHECK-SAME:       outStart = [0, 28, 0],
//CHECK-SAME:       pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}

//CHECK:        PPETask "LRELU"
//CHECK-SAME:       clamp_high = 2147483647 : i64,
//CHECK-SAME:       clamp_low = -2147483648 : i64,
//CHECK-SAME:       fp_prelu_alpha = 1.000000e+00 : f64,
//CHECK-SAME:       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64

//CHECK:        return [[CONV_OUTPUT]] : memref<1x64x56x56xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPool
func @MaxPool(%arg0: memref<1x16x1x4xf16, #NHWC, @CMX_NN>, %arg1 : memref<16x1x1x4xsi32, @CMX_NN>, %arg2 : memref<1x1x1x16xui8, @CMX_NN>)
        -> memref<1x16x1x4xf16, #NHWC, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x16x1x4xf16, #NHWC, @CMX_NN>
        to tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<16x1x1x4xsi32, @CMX_NN>
        to tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>
    %2 = builtin.unrealized_conversion_cast %arg2 : memref<1x1x1x16xui8, @CMX_NN>
        to tensor<1x1x1x16xui8, {mem_space = @CMX_NN}>

    %3 = VPU.NCE.MaxPool(%0, %1, %2) {
                kernel_size = [1, 1],
                pad = {bottom = 0, left = 0, right = 0, top = 0},
                strides = [1, 1],
                activation_window_channel_length = 4
            } -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload [0, 0, 0, 0] [1, 16, 1, 4] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
    }

    %4 = builtin.unrealized_conversion_cast %3 : tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
        to memref<1x16x1x4xf16, #NHWC, @CMX_NN>

    return %4 : memref<1x16x1x4xf16, #NHWC, @CMX_NN>

    // CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x1x4xf16, #NHWC, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:          activation_window_channel_length = 4 : i64,
    // CHECK-SAME:          kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          kernel_size = [1, 1],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = "MAXPOOL"
    // CHECK-SAME:      input(%arg0 : memref<1x16x1x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table(%arg1 : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      activation_window(%arg2 : memref<1x1x1x16xui8, @CMX_NN>)
    // CHECK-SAME:      parent_input(%arg0 : memref<1x16x1x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUT_BUF]] : memref<1x16x1x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x1x4xf16, #NHWC, @CMX_NN>)
    // CHECK:           DPUTask
    // CHECK-SAME:          mpe_mode = "VECTOR_FP16"
    // CHECK-SAME:          outEnd = [3, 0, 15],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConv
func @DepthConv(%arg0: memref<1x16x40x80xf16, #NHWC, @CMX_NN>, %arg1: memref<16x1x4x8xf16, #NHWC, @CMX_NN>,
        %arg2 : memref<16x1x1x4xsi32, @CMX_NN>, %arg3 : memref<1x1x1x16xui8, @CMX_NN>)
        -> memref<1x16x37x73xf16, #NHWC, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x16x40x80xf16, #NHWC, @CMX_NN>
        to tensor<1x16x40x80xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<16x1x4x8xf16, #NHWC, @CMX_NN>
        to tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = builtin.unrealized_conversion_cast %arg2 : memref<16x1x1x4xsi32, @CMX_NN>
        to tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>
    %3 = builtin.unrealized_conversion_cast %arg3 : memref<1x1x1x16xui8, @CMX_NN>
        to tensor<1x1x1x16xui8, {mem_space = @CMX_NN}>

    %4 = VPU.NCE.DepthConvolution(%0, %1, %2, %3) {
                pad = {bottom = 0, left = 0, right = 0, top = 0},
                rawFilterShape = [16, 1, 4, 8],
                strides = [1, 1],
                activation_window_channel_length = 44
            } -> tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload [0, 0, 0, 0] [1, 16, 7, 73] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        VPU.DPU.Workload [0, 0, 7, 0] [1, 16, 7, 73] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        VPU.DPU.Workload [0, 0, 14, 0] [1, 16, 7, 73] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        VPU.DPU.Workload [0, 0, 21, 0] [1, 16, 7, 73] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        VPU.DPU.Workload [0, 0, 28, 0] [1, 16, 9, 73] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
    }

    %5 = builtin.unrealized_conversion_cast %4 : tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>
        to memref<1x16x37x73xf16, #NHWC, @CMX_NN>

    return %5 : memref<1x16x37x73xf16, #NHWC, @CMX_NN>
}

// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x37x73xf16, #NHWC, @CMX_NN>

// CHECK:       VPUIP.NCEClusterTask
// CHECK-SAME:          activation_window_channel_length = 44 : i64,
// CHECK-SAME:          kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
// CHECK-SAME:          kernel_size = [4, 8],
// CHECK-SAME:          kernel_strides = [1, 1],
// CHECK-SAME:          task_type = "DWCONV"
// CHECK-SAME:      input(%arg0 : memref<1x16x40x80xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      weights(%arg1 : memref<16x1x4x8xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      weight_table(%arg2 : memref<16x1x1x4xsi32, @CMX_NN>)
// CHECK-SAME:      activation_window(%arg3 : memref<1x1x1x16xui8, @CMX_NN>)
// CHECK-SAME:      parent_input(%arg0 : memref<1x16x40x80xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      parent_output([[OUT_BUF]] : memref<1x16x37x73xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x37x73xf16, #NHWC, @CMX_NN>)
// CHECK:           DPUTask
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [72, 6, 15],
// CHECK-SAME:          outStart = [0, 0, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [72, 13, 15],
// CHECK-SAME:          outStart = [0, 7, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [72, 20, 15],
// CHECK-SAME:          outStart = [0, 14, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [72, 27, 15],
// CHECK-SAME:          outStart = [0, 21, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [72, 36, 15],
// CHECK-SAME:          outStart = [0, 28, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAdd
func @EltwiseAdd(%arg0: memref<1x64x28x28xf16, #NHWC, @CMX_NN>, %arg1: memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
        -> memref<1x64x28x28xf16, #NHWC, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x64x28x28xf16, #NHWC, @CMX_NN>
        to tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<1x64x28x28xf16, #NHWC, @CMX_NN>
        to tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
                op_type = "ADD",
                ppe = {clamp_high = 2147483647, clamp_low = -2147483648, lrelu_mult = 1, lrelu_shift = 0, mode = "ADD"}
            } -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload [0, 0, 0, 0] [1, 64, 5, 28] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        VPU.DPU.Workload [0, 0, 5, 0] [1, 64, 5, 28] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        VPU.DPU.Workload [0, 0, 10, 0] [1, 64, 5, 28] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        VPU.DPU.Workload [0, 0, 15, 0] [1, 64, 5, 28] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
        VPU.DPU.Workload [0, 0, 20, 0] [1, 64, 8, 28] {bottom = 0, left = 0, right = 0, top = 0} "VECTOR_FP16"
    }

    %3 = builtin.unrealized_conversion_cast %2 : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
        to memref<1x64x28x28xf16, #NHWC, @CMX_NN>

    return %3 : memref<1x64x28x28xf16, #NHWC, @CMX_NN>
}

// CHECK:       [[ALLOC0:%.+]] = memref.alloc() : memref<1x64x28x28xf16, #NHWC, @CMX_NN>

// CHECK:       VPUIP.NCEClusterTask
// CHECK-SAME:          activation_window_channel_length = 0 : i64,
// CHECK-SAME:          task_type = "ELTWISE"
// CHECK-SAME:      input(%arg0 : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      weights(%arg0 : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      parent_input(%arg0 : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      parent_output([[ALLOC0]] : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      outputs([[ALLOC0]] : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
// CHECK:           DPUTask
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [27, 4, 63],
// CHECK-SAME:          outStart = [0, 0, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [27, 9, 63],
// CHECK-SAME:          outStart = [0, 5, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [27, 14, 63],
// CHECK-SAME:          outStart = [0, 10, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [27, 19, 63],
// CHECK-SAME:          outStart = [0, 15, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [27, 27, 63],
// CHECK-SAME:          outStart = [0, 20, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
// CHECK:           PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvWithClusterIdAttr
func @ConvWithClusterIdAttr(%arg0: memref<1x32x16x16xf16, #NHWC, @CMX_NN>, %arg1: memref<64x32x3x3xf16, #NHWC, @CMX_NN>, %arg2 : memref<64x1x1x4xsi32, @CMX_NN>) -> memref<1x64x16x16xf16, #NHWC, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>
        to tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>
        to tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = builtin.unrealized_conversion_cast %arg2 : memref<64x1x1x4xsi32, @CMX_NN>
        to tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

    %3 = VPU.NCE.Convolution(%0, %1, %2) {
                pad = {bottom = 1, left = 1, right = 1, top = 1},
                rawFilterShape = [64, 32, 3, 3],
                ppe = {clamp_high = 2147483647, clamp_low = 0, lrelu_mult = 1, lrelu_shift = 0, mode = "LRELU"},
                strides = [1, 1]
            } -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                 VPU.DPU.Workload [0, 0, 0, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64} "VECTOR_FP16" attributes {cluster_id = 0 : i64}
                 VPU.DPU.Workload [0, 0, 1, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 0 : i64}
                 VPU.DPU.Workload [0, 0, 2, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 0 : i64}
                 VPU.DPU.Workload [0, 0, 3, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 0 : i64}
                 VPU.DPU.Workload [0, 0, 4, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 1 : i64}
                 VPU.DPU.Workload [0, 0, 5, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 1 : i64}
                 VPU.DPU.Workload [0, 0, 6, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 1 : i64}
                 VPU.DPU.Workload [0, 0, 7, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 1 : i64}
                 VPU.DPU.Workload [0, 0, 8, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 2 : i64}
                 VPU.DPU.Workload [0, 0, 9, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 2 : i64}
                 VPU.DPU.Workload [0, 0, 10, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 2 : i64}
                 VPU.DPU.Workload [0, 0, 11, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 2 : i64}
                 VPU.DPU.Workload [0, 0, 12, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 3 : i64}
                 VPU.DPU.Workload [0, 0, 13, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 3 : i64}
                 VPU.DPU.Workload [0, 0, 14, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 3 : i64}
                 VPU.DPU.Workload [0, 0, 15, 0] [1, 64, 1, 16] {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 3 : i64}
    }

    %4 = builtin.unrealized_conversion_cast %3 : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        to memref<1x64x16x16xf16, #NHWC, @CMX_NN>

    return %4 : memref<1x64x16x16xf16, #NHWC, @CMX_NN>
}

// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x16x16xf16, #NHWC, @CMX_NN>

// CHECK:       VPUIP.NCEClusterTask
// CHECK-SAME:          kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
// CHECK-SAME:          kernel_size = [3, 3],
// CHECK-SAME:          kernel_strides = [1, 1],
// CHECK-SAME:          task_type = "CONV"
// CHECK-SAME:      input(%arg0 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      weights(%arg1 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      weight_table(%arg2 : memref<64x1x1x4xsi32, @CMX_NN>)
// CHECK-SAME:      parent_input(%arg0 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      parent_output([[OUT_BUF]] : memref<1x64x16x16xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x64x16x16xf16, #NHWC, @CMX_NN>)
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 0 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 0, 63],
// CHECK-SAME:          outStart = [0, 0, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 0 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 1, 63],
// CHECK-SAME:          outStart = [0, 1, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 0 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 2, 63],
// CHECK-SAME:          outStart = [0, 2, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 0 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 3, 63],
// CHECK-SAME:          outStart = [0, 3, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 1 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 4, 63],
// CHECK-SAME:          outStart = [0, 4, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 1 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 5, 63],
// CHECK-SAME:          outStart = [0, 5, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 1 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 6, 63],
// CHECK-SAME:          outStart = [0, 6, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 1 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 7, 63],
// CHECK-SAME:          outStart = [0, 7, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 2 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 8, 63],
// CHECK-SAME:          outStart = [0, 8, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 2 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 9, 63],
// CHECK-SAME:          outStart = [0, 9, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 2 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 10, 63],
// CHECK-SAME:          outStart = [0, 10, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 2 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 11, 63],
// CHECK-SAME:          outStart = [0, 11, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 3 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 12, 63],
// CHECK-SAME:          outStart = [0, 12, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 3 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 13, 63],
// CHECK-SAME:          outStart = [0, 13, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 3 : i64,
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 14, 63],
// CHECK-SAME:          outStart = [0, 14, 0],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           DPUTask
// CHECK-SAME:          cluster_id = 3 : i64
// CHECK-SAME:          mpe_mode = "VECTOR_FP16",
// CHECK-SAME:          outEnd = [15, 15, 63],
// CHECK-SAME:          outStart = [0, 15, 0],
// CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK:           PPETask "LRELU" {clamp_high = 2147483647 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}

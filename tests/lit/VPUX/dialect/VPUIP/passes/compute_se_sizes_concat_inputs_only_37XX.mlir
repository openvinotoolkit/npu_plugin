//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --compute-se-sizes="only-inputs-concat-over-c=true"  %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NoChangeSOHInput(%input: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                       %weights: !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                       %weights_table: !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                       %output: !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    %conv1_out_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %conv1_out_sm_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %conv1_out:2 = VPUIP.NCEClusterTiling inputs(%input as %arg2: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                                 %weights as %arg3: memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>,
                                                 %weights_table as %arg4: memref<64x1x1x4xsi32, @CMX_NN>)
                                   outputs(%conv1_out_cmx as %arg5: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                           %conv1_out_sm_cmx as %arg6: memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> (!VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
                !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
        %0:2 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
                input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                weights(%arg3 : memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>)
                weight_table(%arg4 : memref<64x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> memref<1x64x56x56xf16, #NHWC, @CMX_NN>, memref<1x64x56x56xi1, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 63], outStart = [0, 28, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
            PPETask "NOOP" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    %conv2_out = VPUIP.NCEClusterTiling inputs(%conv1_out#0 as %arg2: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                               %conv1_out#1 as %arg3: memref<1x64x56x56xi1, #NHWC, @CMX_NN>,
                                               %weights as %arg4: memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>,
                                               %weights_table as %arg5: memref<64x1x1x4xsi32, @CMX_NN>)
                                   outputs(%output as %arg6: memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %0 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
                input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                input_sparsity_map(%arg3 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
                weights(%arg4 : memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>)
                weight_table(%arg5 : memref<64x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_input_sparsity_map(%arg3 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
                parent_output(%arg6 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                outputs(%arg6 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
            -> memref<1x64x56x56xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 63], outStart = [0, 28, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
            PPETask "NOOP" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    return %conv2_out : !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:      VPUIP.NCEClusterTask
    // CHECK-NOT:    input_se_size
    // CHECK-NOT:    output_se_size

    // CHECK:      VPUIP.NCEClusterTask
    // CHECK-NOT:    input_se_size
    // CHECK-NOT:    output_se_size
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SOKInput(%input: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                       %weights: !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                       %weights_table: !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                       %output: !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    %conv1_out_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %conv1_out_sm_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %conv1_out:2 = VPUIP.NCEClusterTiling inputs(%input as %arg2: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                                 %weights as %arg3: memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>,
                                                 %weights_table as %arg4: memref<64x1x1x4xsi32, @CMX_NN>)
                                   outputs(%conv1_out_cmx as %arg5: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                           %conv1_out_sm_cmx as %arg6: memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> (!VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
                !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) {
        %0:2 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
                input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                weights(%arg3 : memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>)
                weight_table(%arg4 : memref<64x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> memref<1x64x56x56xf16, #NHWC, @CMX_NN>, memref<1x64x56x56xi1, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 31], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 63], outStart = [0, 0, 32], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
            PPETask "NOOP" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    %conv2_out = VPUIP.NCEClusterTiling inputs(%conv1_out#0 as %arg2: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                               %conv1_out#1 as %arg3: memref<1x64x56x56xi1, #NHWC, @CMX_NN>,
                                               %weights as %arg4: memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>,
                                               %weights_table as %arg5: memref<64x1x1x4xsi32, @CMX_NN>)
                                   outputs(%output as %arg6: memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
        %0 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
                input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                input_sparsity_map(%arg3 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
                weights(%arg4 : memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>)
                weight_table(%arg5 : memref<64x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_input_sparsity_map(%arg3 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
                parent_output(%arg6 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                outputs(%arg6 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
            -> memref<1x64x56x56xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 31], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 63], outStart = [0, 0, 32], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
            PPETask "NOOP" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    return %conv2_out : !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:      VPUIP.NCEClusterTask
    // CHECK-NOT:    input_se_size
    // CHECK-NOT:    output_se_size

    // CHECK:      VPUIP.NCEClusterTask
    // CHECK:        input_se_size = 32 : i64
    // CHECK-NOT:    output_se_size
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SOHInputsConcatOverC(%input: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                          %weights: !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                          %weights_table: !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                          %output: !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    %conv1_out_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %conv1_out_sm_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %conv1_out:2 = VPUIP.NCEClusterTiling inputs(%input as %arg2: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                                 %weights as %arg3: memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>,
                                                 %weights_table as %arg4: memref<64x1x1x4xsi32, @CMX_NN>)
                                   outputs(%conv1_out_cmx as %arg5: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                           %conv1_out_sm_cmx as %arg6: memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> (!VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
                !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
        %0:2 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
                input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                weights(%arg3 : memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>)
                weight_table(%arg4 : memref<64x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> memref<1x64x56x56xf16, #NHWC, @CMX_NN>, memref<1x64x56x56xi1, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 63], outStart = [0, 28, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
            PPETask "NOOP" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    %conv2_out_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %conv2_out_sm_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %conv2_out:2 = VPUIP.NCEClusterTiling inputs(%input as %arg2: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                                 %weights as %arg3: memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>,
                                                 %weights_table as %arg4: memref<64x1x1x4xsi32, @CMX_NN>)
                                   outputs(%conv2_out_cmx as %arg5: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                           %conv2_out_sm_cmx as %arg6: memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> (!VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
                !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
        %0:2 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
                input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                weights(%arg3 : memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>)
                weight_table(%arg4 : memref<64x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> memref<1x64x56x56xf16, #NHWC, @CMX_NN>, memref<1x64x56x56xi1, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 63], outStart = [0, 28, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
            PPETask "NOOP" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    %concat_out_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %concat_out = VPUIP.ConcatView
        inputs(%conv1_out#0, %conv2_out#0 : !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
                                            !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%concat_out_cmx : !VPUIP.DistributedBuffer<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        -> !VPUIP.DistributedBuffer<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    %concat_sm_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x56x56xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %concat_sm = VPUIP.ConcatView
        inputs(%conv1_out#1, %conv2_out#1 : !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
                                            !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%concat_sm_cmx : !VPUIP.DistributedBuffer<1x128x56x56xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        -> !VPUIP.DistributedBuffer<1x128x56x56xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    %conv3_out = VPUIP.NCEClusterTiling inputs(%concat_out as %arg2: memref<1x128x56x56xf16, #NHWC, @CMX_NN>,
                                               %concat_sm as %arg3: memref<1x128x56x56xi1, #NHWC, @CMX_NN>,
                                               %weights as %arg4: memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>,
                                               %weights_table as %arg5: memref<64x1x1x4xsi32, @CMX_NN>)
                                   outputs(%output as %arg6: memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %0 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
                input(%arg2 : memref<1x128x56x56xf16, #NHWC, @CMX_NN>)
                input_sparsity_map(%arg3 : memref<1x128x56x56xi1, #NHWC, @CMX_NN>)
                weights(%arg4 : memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>)
                weight_table(%arg5 : memref<64x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x128x56x56xf16, #NHWC, @CMX_NN>)
                parent_input_sparsity_map(%arg3 : memref<1x128x56x56xi1, #NHWC, @CMX_NN>)
                parent_output(%arg6 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                outputs(%arg6 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
            -> memref<1x64x56x56xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 63], outStart = [0, 28, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
            PPETask "NOOP" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    return %conv3_out : !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:      VPUIP.NCEClusterTask
    // CHECK-NOT:    input_se_size
    // CHECK-NOT:    output_se_size

    // CHECK:      VPUIP.NCEClusterTask
    // CHECK-NOT:    input_se_size
    // CHECK-NOT:    output_se_size

    // CHECK:      VPUIP.NCEClusterTask
    // CHECK-SAME:   input_se_size = 64 : i64
    // CHECK-NOT:    output_se_size
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SOKInputsConcatOverC(%input: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                           %weights: !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                           %weights_table: !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                           %output: !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    %conv1_out_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %conv1_out_sm_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %conv1_out:2 = VPUIP.NCEClusterTiling inputs(%input as %arg2: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                                 %weights as %arg3: memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>,
                                                 %weights_table as %arg4: memref<64x1x1x4xsi32, @CMX_NN>)
                                   outputs(%conv1_out_cmx as %arg5: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                           %conv1_out_sm_cmx as %arg6: memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> (!VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
                !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) {
        %0:2 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
                input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                weights(%arg3 : memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>)
                weight_table(%arg4 : memref<64x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> memref<1x64x56x56xf16, #NHWC, @CMX_NN>, memref<1x64x56x56xi1, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 31], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 63], outStart = [0, 0, 32], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
            PPETask "NOOP" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    %conv2_out_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %conv2_out_sm_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %conv2_out:2 = VPUIP.NCEClusterTiling inputs(%input as %arg2: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                                 %weights as %arg3: memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>,
                                                 %weights_table as %arg4: memref<64x1x1x4xsi32, @CMX_NN>)
                                   outputs(%conv2_out_cmx as %arg5: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                           %conv2_out_sm_cmx as %arg6: memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> (!VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
                !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) {
        %0:2 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
                input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                weights(%arg3 : memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>)
                weight_table(%arg4 : memref<64x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> memref<1x64x56x56xf16, #NHWC, @CMX_NN>, memref<1x64x56x56xi1, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 31], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 63], outStart = [0, 0, 32], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
            PPETask "NOOP" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    %concat_out_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %concat_out = VPUIP.ConcatView
        inputs(%conv1_out#0, %conv2_out#0 : !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
                                            !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%concat_out_cmx : !VPUIP.DistributedBuffer<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        -> !VPUIP.DistributedBuffer<1x128x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    %concat_sm_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %concat_sm = VPUIP.ConcatView
        inputs(%conv1_out#1, %conv2_out#1 : !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
                                            !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        outputs(%concat_sm_cmx : !VPUIP.DistributedBuffer<1x128x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
        -> !VPUIP.DistributedBuffer<1x128x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    %conv3_out = VPUIP.NCEClusterTiling inputs(%concat_out as %arg2: memref<1x128x56x56xf16, #NHWC, @CMX_NN>,
                                               %concat_sm as %arg3: memref<1x128x56x56xi1, #NHWC, @CMX_NN>,
                                               %weights as %arg4: memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>,
                                               %weights_table as %arg5: memref<64x1x1x4xsi32, @CMX_NN>)
                                   outputs(%output as %arg6: memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
        %0 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
                input(%arg2 : memref<1x128x56x56xf16, #NHWC, @CMX_NN>)
                input_sparsity_map(%arg3 : memref<1x128x56x56xi1, #NHWC, @CMX_NN>)
                weights(%arg4 : memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>)
                weight_table(%arg5 : memref<64x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x128x56x56xf16, #NHWC, @CMX_NN>)
                parent_input_sparsity_map(%arg3 : memref<1x128x56x56xi1, #NHWC, @CMX_NN>)
                parent_output(%arg6 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                outputs(%arg6 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
            -> memref<1x64x56x56xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 31], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 63], outStart = [0, 0, 32], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
            PPETask "NOOP" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    return %conv3_out : !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:      VPUIP.NCEClusterTask
    // CHECK-NOT:    input_se_size
    // CHECK-NOT:    output_se_size

    // CHECK:      VPUIP.NCEClusterTask
    // CHECK-NOT:    input_se_size
    // CHECK-NOT:    output_se_size

    // CHECK:      VPUIP.NCEClusterTask
    // CHECK-SAME:   input_se_size = 32 : i64
    // CHECK-NOT:    output_se_size
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @InputMulitpleVariants(%input: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                            %weights: !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                            %weights_table: !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                            %activation_window: !VPUIP.DistributedBuffer<1x1x1x64xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                            %output: !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    %maxpool_out_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %maxpool_out_sm_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %maxpool_out:2 = VPUIP.NCEClusterTiling inputs(%input as %arg2: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                                   %weights_table as %arg3: memref<64x1x1x4xsi32, @CMX_NN>,
                                                   %activation_window as %arg4: memref<1x1x1x64xui8, @CMX_NN>)
                                          outputs(%maxpool_out_cmx as %arg5: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                                  %maxpool_out_sm_cmx as %arg6: memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> (!VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
                !VPUIP.DistributedBuffer<1x64x56x56xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
        %0:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 27 : i64, kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = "MAXPOOL"}
                input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                weight_table(%arg3 : memref<64x1x1x4xsi32, @CMX_NN>)
                activation_window(%arg4: memref<1x1x1x64xui8, @CMX_NN>)
                parent_input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                output_sparsity_map(%arg6 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
            -> memref<1x64x56x56xf16, #NHWC, @CMX_NN>, memref<1x64x56x56xi1, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [0, 0, 15], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [0, 0, 31], outStart = [0, 0, 16], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [0, 0, 47], outStart = [0, 0, 32], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [0, 0, 63], outStart = [0, 0, 48], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
            PPETask "NOOP" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    %conv2_out = VPUIP.NCEClusterTiling inputs(%maxpool_out#0 as %arg2: memref<1x64x56x56xf16, #NHWC, @CMX_NN>,
                                               %maxpool_out#1 as %arg3: memref<1x64x56x56xi1, #NHWC, @CMX_NN>,
                                               %weights as %arg4: memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>,
                                               %weights_table as %arg5: memref<64x1x1x4xsi32, @CMX_NN>)
                                   outputs(%output as %arg6: memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %0 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
                input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                input_sparsity_map(%arg3 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
                weights(%arg4 : memref<64x64x1x1xf16, {order = #NHWC}, @CMX_NN>)
                weight_table(%arg5 : memref<64x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                parent_input_sparsity_map(%arg3 : memref<1x64x56x56xi1, #NHWC, @CMX_NN>)
                parent_output(%arg6 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
                outputs(%arg6 : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
            -> memref<1x64x56x56xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 27, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [55, 55, 63], outStart = [0, 28, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
            PPETask "NOOP" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
        }
    }

    return %conv2_out : !VPUIP.DistributedBuffer<1x64x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:      VPUIP.NCEClusterTask
    // CHECK-NOT:    input_se_size
    // CHECK-NOT:    output_se_size

    // CHECK:      VPUIP.NCEClusterTask
    // CHECK-SAME:   input_se_size = 16 : i64
    // CHECK-NOT:    output_se_size
}

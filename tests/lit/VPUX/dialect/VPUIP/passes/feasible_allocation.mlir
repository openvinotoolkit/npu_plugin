//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --feasible-allocation="memory-space=CMX_NN second-level-memory-space=DDR" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SimpleGraph
module @SimpleGraph {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x4x4xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x4x4xf16>
    }
// CHECK:   IE.ExecutorResource {{[0-9]+}} of @NCE
// CHECK:   builtin.module @UsedMemory
// CHECK:           IE.MemoryResource 1024 bytes of @CMX_NN

func.func @main(%in: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>, %out: memref<1x16x4x4xf16, #NHWC>) -> memref<1x16x4x4xf16, #NHWC> {
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, [@CMX_NN, 0]> = dense<1> : tensor<1x1x1x16xui8>

    %buf0 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    %t0, %f0 = async.execute -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%in : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%in : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %1 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64}  {
        %2 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, #NHWC>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %3 = VPUIP.Copy inputs(%2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x16x4x4xf16, #NHWC>) -> memref<1x16x4x4xf16, #NHWC>
        async.yield %3 : memref<1x16x4x4xf16, #NHWC>
    }

    %3 = async.await %f3 : !async.value<memref<1x16x4x4xf16, #NHWC>>
    return %3 : memref<1x16x4x4xf16, #NHWC>

    // CHECK:       [[BUF0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[BUF1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[BUF2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.Copy
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SimpleGraphWithReservedMem
module @SimpleGraphWithReservedMem {

module @ReservedMemory {
    module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
    }
}

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x4x4xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x4x4xf16>
    }

// CHECK:   IE.ExecutorResource {{[0-9]+}} of @NCE
// CHECK:   builtin.module @UsedMemory
// CHECK:           IE.MemoryResource 1536 bytes of @CMX_NN

func.func @main(%in: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>, %out: memref<1x16x4x4xf16, #NHWC>) -> memref<1x16x4x4xf16, #NHWC> {
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, [@CMX_NN, 0]> = dense<1> : tensor<1x1x1x16xui8>

    %buf0 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    %t0, %f0 = async.execute -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%in : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%in : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %1 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64}  {
        %2 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2 : !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x4x4xf16, #NHWC>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %3 = VPUIP.Copy inputs(%2 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x16x4x4xf16, #NHWC>) -> memref<1x16x4x4xf16, #NHWC>
        async.yield %3 : memref<1x16x4x4xf16, #NHWC>
    }

    %3 = async.await %f3 : !async.value<memref<1x16x4x4xf16, #NHWC>>
    return %3 : memref<1x16x4x4xf16, #NHWC>

    // CHECK:       [[BUF0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[BUF1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[BUF2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.Copy
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoOutputs
module @TwoOutputs {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x4x4xf16>
    }
    outputsInfo : {
        DataInfo "prob1" : tensor<1x16x4x4xf16>
        DataInfo "prob2" : tensor<1x16x4x4xf16>
    }

// CHECK:   IE.ExecutorResource {{[0-9]+}} of @NCE
// CHECK:   builtin.module @UsedMemory
// CHECK:           IE.MemoryResource 1024 bytes of @CMX_NN

func.func @main(%arg0: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x16x4x4xf16, #NHWC>, %arg2: memref<1x16x4x4xf16, #NHWC>)
        -> (memref<1x16x4x4xf16, #NHWC>, memref<1x16x4x4xf16, #NHWC>) {
    %cst = const.Declare memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]> = dense<1.000000e+00> : tensor<1x16x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, [@CMX_NN, 0]> = dense<1> : tensor<1x1x1x16xui8>

    %buf0 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    %token, %results = async.execute -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64}  {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%arg0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %token_0, %results_1 = async.execute -> !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>> 
            attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64}  {
        %1 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
            input(%cst : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%cst : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 4, 4], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %token_2, %results_3 = async.execute [%token] (%results as %arg3: !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>) -> !async.value<memref<1x16x4x4xf16, #NHWC>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64}  {
        %2 = VPUIP.Copy inputs(%arg3 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x16x4x4xf16, #NHWC>) -> memref<1x16x4x4xf16, #NHWC>
        async.yield %2 : memref<1x16x4x4xf16, #NHWC>
    }

    %token_4, %results_5 = async.execute [%token_0] (%results_1 as %arg3: !async.value<memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>>) -> !async.value<memref<1x16x4x4xf16, #NHWC>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64}  {
        %3 = VPUIP.Copy inputs(%arg3 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x16x4x4xf16, #NHWC>) -> memref<1x16x4x4xf16, #NHWC>
        async.yield %3 : memref<1x16x4x4xf16, #NHWC>
    }

    %4 = async.await %results_3 : !async.value<memref<1x16x4x4xf16, #NHWC>>
    %5 = async.await %results_5 : !async.value<memref<1x16x4x4xf16, #NHWC>>
    return %4, %5 : memref<1x16x4x4xf16, #NHWC>, memref<1x16x4x4xf16, #NHWC>

    // CHECK:       [[BUF0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[BUF1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPUIP.Copy

    // CHECK:       VPUIP.Copy
}

}

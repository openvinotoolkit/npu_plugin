//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --feasible-allocation="memory-space=CMX_NN second-level-memory-space=DDR" %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Spilling
module @Spilling {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x96x96xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x96x96xf16>
    }

func.func @main(%in: memref<1x16x96x96xf16, #NHWC>, %out: memref<1x16x96x96xf16, #NHWC>) -> memref<1x16x96x96xf16, #NHWC> {
    %cst0 = const.Declare memref<1x16x96x96xf16, #NHWC> = dense<2.0> : tensor<1x16x96x96xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, [@CMX_NN, 0]> = dense<1> : tensor<1x1x1x16xui8>

    %buf_in = memref.alloc() : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>

    %buf0 = memref.alloc() : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>

    %t_in, %r_in = async.execute -> !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.Copy inputs(%in : memref<1x16x96x96xf16, #NHWC>) outputs(%buf_in : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t0, %r0 = async.execute -> !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.Copy inputs(%cst0 : memref<1x16x96x96xf16, #NHWC>) outputs(%buf0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t3, %r3 = async.execute [%t_in] (%r_in as %0 : !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %1 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "MAXPOOL"
            }
            input(%0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t1, %r1 = async.execute [%t3, %t0] (%r3 as %0 : !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>, %r0 as %1 : !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %2 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%1 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %2 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t5, %r5 = async.execute [%t_in, %t1] (%r_in as %0 : !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>, %r1 as %1 : !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %2 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%1 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf3 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf3 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %2 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t6, %r6 = async.execute [%t5] (%r5 as %0 : !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x96x96xf16, #NHWC>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64} {
        %1 = VPUIP.Copy inputs(%0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x16x96x96xf16, #NHWC>) -> memref<1x16x96x96xf16, #NHWC>
        async.yield %1 : memref<1x16x96x96xf16, #NHWC>
    }

    %6 = async.await %r6 : !async.value<memref<1x16x96x96xf16, #NHWC>>
    return %6 : memref<1x16x96x96xf16, #NHWC>

    // CHECK:       [[BUF_SPILL_WRITE:%.*]] = memref.alloc() : memref<1x16x96x96xf16, #NHWC, @DDR>
    // CHECK:       [[BUF_SPILL_READ:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <
    // CHECK-SAME:      > -> memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT:       VPUIP.Copy inputs(%arg0 : memref<1x16x96x96xf16, #NHWC>) outputs([[BUF0:%.*]] :

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.NCEClusterTask
    // CHECK-SAME:         task_type = "MAXPOOL"

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.Copy
    // CHECK-SAME:          inputs([[BUF0]] : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) outputs([[BUF_SPILL_WRITE]] : memref<1x16x96x96xf16, #NHWC, @DDR>)

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.Copy

    // CHECK:       [[T22:%.+]], [[R22:%.+]] = async.execute
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = "ELTWISE"

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-SAME:      ([[R2]] as [[ARG1:%.*]]: !async.value<memref<1x16x96x96xf16, #NHWC, @DDR>>
    // CHECK:       VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG1:%.*]] : memref<1x16x96x96xf16, #NHWC, @DDR>) outputs([[BUF_SPILL_READ]] : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK-SAME:      ([[R4]] as [[ARG2:%.*]]: !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>,
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = "ELTWISE"
    // CHECK-SAME:      input([[ARG2]] : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.Copy
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SpillingOpWith2Outputs
module @SpillingOpWith2Outputs {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x96x96xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x96x96xf16>
    }

func.func @main(%in: memref<1x16x96x96xf16, #NHWC>, %out: memref<1x16x96x96xf16, #NHWC>) -> memref<1x16x96x96xf16, #NHWC> {
    %cst0 = const.Declare memref<1x16x96x96xf16, #NHWC> = dense<2.0> : tensor<1x16x96x96xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, [@CMX_NN, 0]> = dense<1> : tensor<1x1x1x16xui8>

    %buf0 = memref.alloc() : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    %buf4 = memref.alloc() : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    %buf5 = memref.alloc() : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>

    %t0, %r0 = async.execute -> !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.Copy inputs(%in : memref<1x16x96x96xf16, #NHWC>) outputs(%buf0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t1, %r1 = async.execute -> !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.Copy inputs(%cst0 : memref<1x16x96x96xf16, #NHWC>) outputs(%buf3 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    }

    // Below operation has two outputs and one of them would need to be spilled when scheduled
    %t2, %r2:2 = async.execute [%t0] (%r0 as %arg0 : !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>)
            -> (!async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>, !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>)
            attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %1 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "MAXPOOL"
            }
            input(%arg0: memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%arg0: memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        %2 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "MAXPOOL"
            }
            input(%arg0: memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%arg0: memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1, %2 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t3, %r3 = async.execute [%t1, %t2] (%r2#0 as %arg0 : !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>, %r1 as %arg1 : !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%arg0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%arg1 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf4 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf4 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t4, %r4 = async.execute [%t1, %t3] (%r2#1 as %arg0 : !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>, %r3 as %arg1 : !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%arg0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%arg1 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf5 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf5 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t5, %r5 = async.execute [%t4] (%r4 as %arg0 : !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x96x96xf16, #NHWC>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64} {
        %0 = VPUIP.Copy inputs(%arg0 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x16x96x96xf16, #NHWC>) -> memref<1x16x96x96xf16, #NHWC>
        async.yield %0 : memref<1x16x96x96xf16, #NHWC>
    }

    %3 = async.await %r5 : !async.value<memref<1x16x96x96xf16, #NHWC>>
    return %3 : memref<1x16x96x96xf16, #NHWC>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT       VPUIP.Copy
    // CHECK:       [[T1:%.+]], [[R1:%.+]]:2 = async.execute {{.*}} ([[R0]] as %arg2: !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>)
    // CHECK-NEXT       VPUIP.NCEClusterTask
    // CHECK-SAME        task_type = "MAXPOOL"
    // CHECK-SAME        inputs(%arg2 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) outputs([[BUF1:%.*]] :
    // CHECK-NEXT       VPUIP.NCEClusterTask
    // CHECK-SAME        task_type = "MAXPOOL"
    // CHECK-SAME        inputs(%arg2 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>) outputs([[BUF2:%.*]] :
    // CHECK:       [[T_SPILL_WRITE:%.+]], [[R_SPILL_WRITE:%.+]] = async.execute
    // CHECK-NEXT       VPUIP.Copy inputs([[BUF2]] : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-NEXT       VPUIP.Copy
    // CHECK-SAME       -> memref<1x16x96x96xf16, #NHWC, @DDR>
    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute {{.*}} ([[R1]]#0 as %arg2: !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>
    // CHECK-NEXT       VPUIP.NCEClusterTask
    // CHECK-SAME         task_type = "ELTWISE"
    // CHECK-SAME         inputs(%arg2 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:       [[T_SPILL_READ:%.+]], [[R_SPILL_READ:%.+]] = async.execute {{.*}} ([[R_SPILL_WRITE]] as %arg2: !async.value<memref<1x16x96x96xf16, #NHWC, @DDR>>)
    // CHECK-NEXT       VPUIP.Copy inputs(%arg2 : memref<1x16x96x96xf16, #NHWC, @DDR>)
    // CHECK-SAME       -> memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute {{.*}} ([[R_SPILL_READ]] as %arg2: !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>
    // CHECK-NEXT       VPUIP.NCEClusterTask
    // CHECK-SAME         task_type = "ELTWISE"
    // CHECK-SAME         inputs(%arg2 : memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute {{.*}} ([[R6]] as %arg2: !async.value<memref<1x16x96x96xf16, #NHWC, [@CMX_NN, 0]>>
    // CHECK-NEXT       VPUIP.Copy
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#strides = [131072, 1, 2048, 32]

// CHECK-LABEL: @SpillingOfSubViewBuffer
module @SpillingOfSubViewBuffer {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x64x64xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x64x64xf16>
    }

func.func @main(%in: memref<1x16x64x64xf16, #NHWC>, %out: memref<1x16x64x64xf16, #NHWC>) -> memref<1x16x64x64xf16, #NHWC> {
    %cst0 = const.Declare memref<1x16x64x64xf16, #NHWC> = dense<2.0> : tensor<1x16x64x64xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, [@CMX_NN, 0]> = dense<1> : tensor<1x1x1x16xui8>

    // master buffer that will get spilled
    %buf_master = memref.alloc() : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>

    %buf0 = memref.alloc() : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    %buf4 = memref.alloc() : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    %buf5 = memref.alloc() : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    %buf6 = memref.alloc() : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>

    %t_dma_in, %r_dma_in = async.execute -> !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.Copy inputs(%in : memref<1x16x64x64xf16, #NHWC>) outputs(%buf0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    // Operation that is using master buffer which will not be directly identified for spilling but for
    // which dependant operations still need to be updated as it uses spilled master buffer
    %t0, %r0 = async.execute [%t_dma_in] (%r_dma_in as %arg0 : !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.SubView %buf_master [0, 16, 0, 0][1, 16, 64, 64] : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]> to memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
        %1 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "MAXPOOL"
            }
            input(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 64, 64], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    // Operation that is using master buffer and will be identified as necessary for spilling
    // Dependant operations will need to be updated to refer to spillRead result
    %t1, %r1 = async.execute [%t0] (%r0 as %arg0 : !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %0 = VPUIP.SubView %buf_master [0, 0, 0, 0][1, 16, 64, 64] : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]> to memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
        %1 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "MAXPOOL"
            }
            input(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 64, 64], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    %t2, %r2 = async.execute -> !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %0 = VPUIP.Copy inputs(%cst0 : memref<1x16x64x64xf16, #NHWC>) outputs(%buf1 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    %t3, %r3 = async.execute [%t1] (%r1 as %arg0 : !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "MAXPOOL"
            }
            input(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 64, 64], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    %t4, %r4 = async.execute [%t3, %t2] (%r3 as %arg0 : !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>, %r2 as %arg1 : !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64} {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weights(%arg1 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%buf3 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%buf3 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 64, 64], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    // operation that is using buffer that will be spilled through result of async exec op
    %t5, %r5 = async.execute [%t1, %t4] (%r1 as %arg0 : !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>, %r4 as %arg1 : !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 6 : i64} {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weights(%arg1 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%buf4 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%buf4 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 64, 64], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    // operation that is using directly master buffer that will be spilled
    %t6, %r6 = async.execute [%t5] (%r5 as %arg0 : !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 7 : i64} {
        %0 = VPUIP.SubView %buf_master [0, 0, 0, 0][1, 16, 64, 64] : memref<1x32x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]> to memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
        %1 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weights(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%buf5 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%buf5 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 64, 64], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %1 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    // operation that is a user of other op that is also using master buffer which got spilled
    %t7, %r7 = async.execute [%t6] (%r0 as %arg0 : !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>, %r6 as %arg1 : !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 8 : i64} {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            weights(%arg1 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            parent_output(%buf6 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>)
            outputs(%buf6 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 64, 64], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>
    }

    %t_tma_out, %r_dma_out = async.execute [%t7] (%r7 as %arg0 : !async.value<memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x64x64xf16, #NHWC>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 9 : i64} {
        %0 = VPUIP.Copy inputs(%arg0 : memref<1x16x64x64xf16, {order = #NHWC, strides = #strides}, [@CMX_NN, 0]>) outputs(%out : memref<1x16x64x64xf16, #NHWC>) -> memref<1x16x64x64xf16, #NHWC>
        async.yield %0 : memref<1x16x64x64xf16, #NHWC>
    }

    %result = async.await %r_dma_out : !async.value<memref<1x16x64x64xf16, #NHWC>>
    return %result : memref<1x16x64x64xf16, #NHWC>

    // CHECK:       [[BUF_MASTER:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <
    // CHECK-SAME:      > -> memref<1x32x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>
    // CHECK:       [[BUF_SPILL_WRITE:%.*]] = memref.alloc() : memref<1x32x64x64xf16, #NHWC, @DDR>
    // CHECK:       [[BUF_SPILL_READ:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <
    // CHECK-SAME:      > -> memref<1x32x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK:       VPUIP.Copy

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK:       VPUIP.SubView
    // CHECK-SAME:      [0, 16, 0, 0] [1, 16, 64, 64] : memref<1x32x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]> to memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = "MAXPOOL"

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK:       VPUIP.SubView
    // CHECK-SAME:      [0, 0, 0, 0] [1, 16, 64, 64] : memref<1x32x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]> to memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = "MAXPOOL"

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = "MAXPOOL"

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK:       VPUIP.Copy
    // CHECK-SAME:      spillId
    // CHECK-SAME:      inputs([[BUF_MASTER]] : memref<1x32x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>) outputs([[BUF_SPILL_WRITE]] : memref<1x32x64x64xf16, #NHWC, @DDR>)

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK:       VPUIP.Copy

    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = "ELTWISE"

    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute
    // CHECK-SAME:      ([[R4]] as [[ARG0:%.*]]: !async.value<memref<1x32x64x64xf16, #NHWC, @DDR>>
    // CHECK:       VPUIP.Copy
    // CHECK-SAME:      spillId
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x32x64x64xf16, #NHWC, @DDR>) outputs([[BUF_SPILL_READ]] : memref<1x32x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>)

    // CHECK:       [[T8:%.+]], [[R8:%.+]] = async.execute
    // CHECK-SAME:      ([[R7]] as [[ARG1:%.*]]: !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>>
    // CHECK:       [[SUBVIEW_0:%.*]] = VPUIP.SubView [[ARG1]] [0, 0, 0, 0] [1, 16, 64, 64] : memref<1x32x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]> to memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = "ELTWISE"
    // CHECK-SAME:      input([[SUBVIEW_0]] : memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>

    // CHECK:       [[T9:%.+]], [[R9:%.+]] = async.execute
    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView [[BUF_SPILL_READ]] [0, 0, 0, 0] [1, 16, 64, 64] : memref<1x32x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]> to memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = "ELTWISE"
    // CHECK-SAME:      input([[SUBVIEW_1]] : memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>

    // CHECK:       [[T10:%.+]], [[R10:%.+]] = async.execute
    // CHECK-SAME:      ([[R7]] as [[ARG2:%.*]]: !async.value<memref<1x32x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>>
    // CHECK:       [[SUBVIEW_2:%.*]] = VPUIP.SubView [[ARG2]] [0, 16, 0, 0] [1, 16, 64, 64] : memref<1x32x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]> to memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = "ELTWISE"
    // CHECK-SAME:      input([[SUBVIEW_2]] : memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, [@CMX_NN, 0]>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SpillWriteOptimize
module @SpillWriteOptimize {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x85x85xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x85x85xf16>
    }

func.func @main(%in: memref<1x16x85x85xf16, #NHWC>, %out: memref<1x16x85x85xf16, #NHWC>) -> memref<1x16x85x85xf16, #NHWC> {
    %cst0 = const.Declare memref<1x16x85x85xf16, #NHWC> = dense<2.0> : tensor<1x16x85x85xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, [@CMX_NN, 0]> = dense<1> : tensor<1x1x1x16xui8>

    %buf_in = memref.alloc() : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>

    %buf0 = memref.alloc() : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    %buf4 = memref.alloc() : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    %buf5 = memref.alloc() : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    %buf6 = memref.alloc() : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>

    %t_in, %r_in = async.execute -> !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.Copy inputs(%in : memref<1x16x85x85xf16, #NHWC>) outputs(%buf_in : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t0, %r0 = async.execute -> !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.Copy inputs(%cst0 : memref<1x16x85x85xf16, #NHWC>) outputs(%buf0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t1, %r1 = async.execute [%t_in] (%r_in as %0 : !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %1 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "MAXPOOL"
            }
            input(%0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 85, 85], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t2, %r2 = async.execute [%t0, %t1] (%r0 as %0 : !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>>, %r1 as %1 : !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %2 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%1 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 85, 85], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %2 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t3, %r3 = async.execute [%t_in, %t2] (%r_in as %0 : !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>>, %r2 as %1 : !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %2 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%1 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf3 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf3 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 85, 85], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %2 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t4, %r4 = async.execute [%t3] -> !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64} {
        %0 = VPUIP.Copy inputs(%cst0 : memref<1x16x85x85xf16, #NHWC>) outputs(%buf4 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t5, %r5 = async.execute [%t3, %t4] (%r3 as %0 : !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>>, %r4 as %1 : !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 6 : i64} {
        %2 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%1 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf5 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf5 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 85, 85], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %2 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t6, %r6 = async.execute [%t_in, %t5] (%r_in as %0 : !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>>, %r5 as %1 : !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 7 : i64} {
        %2 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%1 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf6 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf6 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 85, 85], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %2 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t7, %r7 = async.execute [%t6] (%r6 as %0 : !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x85x85xf16, #NHWC>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 8 : i64} {
        %1 = VPUIP.Copy inputs(%0 : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x16x85x85xf16, #NHWC>) -> memref<1x16x85x85xf16, #NHWC>
        async.yield %1 : memref<1x16x85x85xf16, #NHWC>
    }

    %result = async.await %r7 : !async.value<memref<1x16x85x85xf16, #NHWC>>
    return %result : memref<1x16x85x85xf16, #NHWC>

    // CHECK:       [[BUF_SPILL_WRITE:%.*]] = memref.alloc() : memref<1x16x85x85xf16, #NHWC, @DDR>
    // CHECK:       [[BUF_SPILL_READ0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <
    // CHECK:       [[BUF_SPILL_READ1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <

    // Operation 0 whose output will be later spilled
    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT:       VPUIP.Copy inputs(%arg0 : memref<1x16x85x85xf16, #NHWC>) outputs([[BUF_TO_SPILL:%.*]] :

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = "MAXPOOL"

    // First SPILL WRITE for buffer from operation 0
    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.Copy
    // CHECK-SAME:          inputs([[BUF_TO_SPILL]] : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>) outputs([[BUF_SPILL_WRITE]] : memref<1x16x85x85xf16, #NHWC, @DDR>)

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.Copy

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = "ELTWISE"

    // First SPILL READ of spilled buffer from operation 0
    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK-SAME:      ([[R2]] as [[ARG0:%.*]]: !async.value<memref<1x16x85x85xf16, #NHWC, @DDR>>
    // CHECK:       VPUIP.Copy
    // CHECK-SAME:      spillId
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x16x85x85xf16, #NHWC, @DDR>) outputs([[BUF_SPILL_READ0]] : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       [[T6:%.+]], [[R6:%.+]] = async.execute
    // CHECK-SAME:      ([[R5]] as [[ARG1:%.*]]: !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>>,
    // CHECK-NEXT:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = "ELTWISE"
    // CHECK-SAME:      input([[ARG1]] : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[T7:%.+]], [[R7:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.Copy

    // Here second SPILL WRITE of operation 0 output would be inserted if no optimization was performed

    // CHECK:       [[T8:%.+]], [[R8:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = "ELTWISE"

    // Second SPILL READ of spilled buffer from operation 0
    // CHECK:       [[T9:%.+]], [[R9:%.+]] = async.execute
    // CHECK-SAME:      ([[R2]] as [[ARG2:%.*]]: !async.value<memref<1x16x85x85xf16, #NHWC, @DDR>>
    // CHECK:       VPUIP.Copy
    // CHECK-SAME:      spillId
    // CHECK-SAME:      inputs([[ARG2]] : memref<1x16x85x85xf16, #NHWC, @DDR>) outputs([[BUF_SPILL_READ1]] : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       [[T10:%.+]], [[R10:%.+]] = async.execute
    // CHECK-SAME:      ([[R9]] as [[ARG3:%.*]]: !async.value<memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>>,
    // CHECK-NEXT:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = "ELTWISE"
    // CHECK-SAME:      input([[ARG3]] : memref<1x16x85x85xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[T11:%.+]], [[R11:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.Copy
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ControlEdgeOverlapMemory
module @ControlEdgeOverlapMemory {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x112x112xf16>
    }
    outputsInfo : {
        DataInfo "prob0" : tensor<1x16x112x112xf16>
        DataInfo "prob1" : tensor<1x16x112x112xf16>
    }
func.func @main(%in: memref<1x16x112x112xf16, #NHWC>, %out0: memref<1x16x112x112xf16, #NHWC>, %out1: memref<1x16x112x112xf16, #NHWC>) -> (memref<1x16x112x112xf16, #NHWC>, memref<1x16x112x112xf16, #NHWC>) {
    %wt = const.Declare memref<16x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, [@CMX_NN, 0]> = dense<1> : tensor<1x1x1x16xui8>

    %buf0 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>

    // Task 0
    %t0, %f0 = async.execute -> !async.value<memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.Copy inputs(%in : memref<1x16x112x112xf16, #NHWC>) outputs(%buf0 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>
        async.yield %0 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>
    }

    // Task 1
    %t1, %f1 = async.execute (%f0 as %arg0 : !async.value<memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "MAXPOOL"
            }
            input(%arg0 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf1 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf1 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 112, 112], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>
    }

    // Task 2
    %t2, %f2 = async.execute (%f1 as %arg0 : !async.value<memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x112x112xf16, #NHWC>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %0 = VPUIP.Copy inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out0 : memref<1x16x112x112xf16, #NHWC>) -> memref<1x16x112x112xf16, #NHWC>
        async.yield %0 : memref<1x16x112x112xf16, #NHWC>
    }

    // Task 3
    %t3, %f3 = async.execute (%f0 as %arg0 : !async.value<memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %0 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "MAXPOOL"
            }
            input(%arg0 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%arg0 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 112, 112], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %0 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>
    }

    // Task 4
    %t4, %f4 = async.execute (%f3 as %arg0 : !async.value<memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x16x112x112xf16, #NHWC>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %0 = VPUIP.Copy inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out1 : memref<1x16x112x112xf16, #NHWC>) -> memref<1x16x112x112xf16, #NHWC>
        async.yield %0 : memref<1x16x112x112xf16, #NHWC>
    }

    %r0 = async.await %f2 : !async.value<memref<1x16x112x112xf16, #NHWC>>
    %r1 = async.await %f4 : !async.value<memref<1x16x112x112xf16, #NHWC>>
    return %r0, %r1 : memref<1x16x112x112xf16, #NHWC>, memref<1x16x112x112xf16, #NHWC>

    // Token dependencies will match data flow by default:
    //  Task0 -> Task1 -> Task2
    //  Task0 -> Task3 -> Task4
    // besides that due to overlapping memory ranges of Task3 and Task1
    // additional control edge will be inserted:
    //  Task2 -> Task3
    // Optimization of token dependencies (transitive reduction) is beyond
    // this pass and done as a separate step

    // CHECK:       [[BUF0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <401408>
    // CHECK:       [[BUF1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0>
    // CHECK:       [[BUF2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT:      VPUIP.Copy
    // CHECK-SAME:      outputs([[BUF0]]

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-SAME:      [[T0]]
    // CHECK-NEXT:      VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]]

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-SAME:      [[T0]], [[T1]]
    // CHECK-NEXT:      VPUIP.Copy

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-SAME:      [[T0]], [[T1]], [[T2]]
    // CHECK-NEXT:      VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF2]]

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-SAME:      [[T2]], [[T3]]
    // CHECK-NEXT:      VPUIP.Copy
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ControlEdgeOverlapMemoryCheckProdCons
module @ControlEdgeOverlapMemoryCheckProdCons {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x32x60x60xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x32x60x60xf16>
    }

func.func @main(%in: memref<1x32x60x60xf16, #NHWC>, %out: memref<1x32x60x60xf16, #NHWC>) -> memref<1x32x60x60xf16, #NHWC> {
    %cst0 = const.Declare memref<1x32x60x60xf16, #NHWC> = dense<2.0> : tensor<1x32x60x60xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare memref<32x1x1x4xsi32, [@CMX_NN, 0]> = dense<1> : tensor<32x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x32xui8, [@CMX_NN, 0]> = dense<1> : tensor<1x1x1x32xui8>

    %buf_in = memref.alloc() : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>

    %buf0 = memref.alloc() : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>

    %t_in, %r_in = async.execute -> !async.value<memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.Copy inputs(%in : memref<1x32x60x60xf16, #NHWC>) outputs(%buf_in : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>
        async.yield %0 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t3, %r3 = async.execute [%t_in] (%r_in as %0 : !async.value<memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %1 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 27 : i64,
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "MAXPOOL"
            }
            input(%0 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%wt : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_win : memref<1x1x1x32xui8, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf0 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf0 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
            }
        async.yield %1 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t0, %r0 = async.execute -> !async.value<memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %0 = VPUIP.Copy inputs(%cst0 : memref<1x32x60x60xf16, #NHWC>) outputs(%buf1 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>
        async.yield %0 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t1, %r1 = async.execute [%t3, %t0] (%r3 as %0 : !async.value<memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>>, %r0 as %1 : !async.value<memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %2 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%0 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%1 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf2 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf2 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %2 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t5, %r5 = async.execute [%t_in, %t1] (%r_in as %0 : !async.value<memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>>, %r1 as %1 : !async.value<memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %2 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE"
            }
            input(%0 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%1 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%0 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf3 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf3 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>
            variants :
            {
                DPUTask { outEnd = [16, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
        async.yield %2 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>
    }

    %t6, %r6 = async.execute [%t5] (%r5 as %0 : !async.value<memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x32x60x60xf16, #NHWC>> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 5 : i64} {
        %1 = VPUIP.Copy inputs(%0 : memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x32x60x60xf16, #NHWC>) -> memref<1x32x60x60xf16, #NHWC>
        async.yield %1 : memref<1x32x60x60xf16, #NHWC>
    }

    %6 = async.await %r6 : !async.value<memref<1x32x60x60xf16, #NHWC>>
    return %6 : memref<1x32x60x60xf16, #NHWC>

    // Token dependencies will match data flow by default:
    //  Task0 -> Task1 -> Task3 -> Task4
    //  Task2 -> Task3
    //  Task0 -> Task_SW -> Task_SR -> Task4 -> Task5
    // besides that due to overlapping memory ranges additional control edge will be inserted.
    // Important is relation between Task0, Task1, Task_SW.
    // Execution order is following:
    //  t0: Task0 produces BUF0
    //  tX: Task1 reads BUF0
    //  tY: Task_SW reads BUF0
    // Resulting dependencies from just looking at memory intervals and their users throughout execution time
    // is following: Task0 -> Task1, Task0 -> Task_SW
    // If there would be no differentiation between resource producer and consumer unnecessary dependency
    // would be inserted from Task1 -> Task_SW
    //
    // Optimization of token dependencies (transitive reduction) is beyond
    // this pass and done as a separate step

    // CHECK:       [[BUF0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0>
    // CHECK:       [[BUF1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <230400>
    // CHECK:       [[BUF2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0>
    // CHECK:       [[BUF3:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <460800>
    // CHECK:       [[BUF4:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <230400>
    // CHECK:       [[BUF_SPILL_WRITE:%.*]] = memref.alloc() : memref<1x32x60x60xf16, #NHWC, @DDR>
    // CHECK:       [[BUF_SPILL_READ:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x32x60x60xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[T0:%.+]], [[R0:%.+]] = async.execute ->
    // CHECK-NEXT:       VPUIP.Copy
    // CHECK-SAME:       outputs([[BUF0]]

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute
    // CHECK-SAME:       [[T0]]
    // CHECK-NEXT:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = "MAXPOOL"
    // CHECK-SAME:       outputs([[BUF1]]

    // CHECK:       [[T_SW:%.+]], [[R_SW:%.+]] = async.execute
    // CHECK-SAME:       [[T0]]
    // CHECK-NOT:        [[T1]]
    // CHECK-NEXT:       VPUIP.Copy
    // CHECK-SAME:       inputs([[BUF0]]
    // CHECK-SAME:       outputs([[BUF_SPILL_WRITE]]

    // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
    // CHECK-NEXT:       VPUIP.Copy
    // CHECK-SAME:       outputs([[BUF2]]

    // CHECK:       [[T3:%.+]], [[R3:%.+]] = async.execute
    // CHECK-SAME:       [[T1]], [[T2]]
    // CHECK-NEXT:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = "ELTWISE"
    // CHECK-SAME:       outputs([[BUF3]]

    // CHECK:       [[T_SR:%.+]], [[R_SR:%.+]] = async.execute
    // CHECK-SAME:       [[T3]], [[T_SW]]
    // CHECK-NEXT:       VPUIP.Copy
    // CHECK-SAME:       outputs([[BUF_SPILL_READ]]

    // CHECK:       [[T4:%.+]], [[R4:%.+]] = async.execute
    // CHECK-SAME:       [[T0]], [[T1]], [[T3]], [[T_SR]]
    // CHECK-NEXT:       VPUIP.NCEClusterTask
    // CHECK-SAME:       task_type = "ELTWISE"
    // CHECK-SAME:       outputs([[BUF4]]

    // CHECK:       [[T5:%.+]], [[R5:%.+]] = async.execute
    // CHECK-SAME:       [[T4]]
    // CHECK-NEXT:       VPUIP.Copy
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = {bottom = 1, left = 1, right = 1, top = 1},
    num_clusters = 4
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Input_DDR = memref<1x32x16x16xf16, #NHWC, @DDR>
!Weights_DDR = memref<64x32x3x3xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<64x1x1x4xsi32, #NHWC, @DDR>
!Output_DDR = memref<1x64x16x16xf16, #NHWC, @DDR>

!WeightsTableStub = memref<64x1x1x4xsi32>
!InputStub_CMX = memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = memref<64x32x3x3xf16, #NHWC, [@CMX_NN, 0]>
!WeightsTableStub_CMX = memref<64x1x1x4xsi32, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @SingleConvWithClustering
module @SingleConvWithClustering {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x32x16x16xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x64x16x16xf16>
    }

func.func @main(%input: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare memref<64x32x3x3xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %output_buff_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %t0 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !Input_DDR) outputs(%input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
            %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
        }

        async.yield
    }

    %t1 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%weights as %arg0: !Weights_DDR) outputs(%weights_cmx as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
            %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
        }

        async.yield
    }

    %t2 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%weights_table as %arg0: !WeightsTable_DDR) outputs(%weights_table_cmx as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
            %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
        }

        async.yield
    }

    %t3 = async.execute [%t0, %t1, %t2]
                attributes {VPUIP.executor = @NCE, VPUIP.num_units = 4 : i64, "async-deps-index" = 3 : i64} {
            %0 = VPUIP.NCEClusterTiling
                    inputs(%input_cmx as %arg0: !InputStub_CMX,
                            %weights_cmx as %arg1: !WeightsStub_CMX,
                            %weights_table_cmx as %arg2: !WeightsTableStub_CMX)
                    outputs(%output_buff_cmx as %arg3: !OutputStub_CMX)
                        -> !OutputStub_CMX {

                  %1 = VPUIP.NCEClusterTask {
                            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                            kernel_size = [1, 1],
                            kernel_strides = [1, 1],
                            task_type = "CONV"
                        }  input(%arg0 : !InputStub_CMX)
                            weights(%arg1 : !WeightsStub_CMX)
                            weight_table(%arg2 : !WeightsTableStub_CMX)
                            parent_input(%arg0 : !InputStub_CMX)
                            parent_output(%arg3 : !OutputStub_CMX)
                            outputs(%arg3 : !OutputStub_CMX)
                                -> !OutputStub_CMX variants :  {
                            DPUTask {
                                outStart = [0, 0, 0], outEnd = [31, 15, 15],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
                            }
                            } PPE :  {
                            }
            }

            async.yield
    }

    %t4 = async.execute [%t3]
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 4 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%output_buff_cmx as %arg0: !OutputStub_CMX) outputs(%output as %arg1: !Output_DDR) -> !Output_DDR {
            %1 = VPUIP.Copy { out_mem_space = @DDR } inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
        }

        async.yield
    }

    return %output: !Output_DDR


    // CHECK-DAG:       [[CST_WEIGHTS:%.*]] = const.Declare memref<64x32x3x3xf16, #NHWC, @DDR>
    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.*]] = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR>
    // CHECK:       [[BUF0:%.*]] = VPURT.DeclareBuffer "CMX_NN" <45056> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF1:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF2:%.*]] = VPURT.DeclareBuffer "CMX_NN" <49152> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF3:%.*]] = VPURT.DeclareBuffer "CMX_NN" <36864> -> !VPUIP.DistributedBuffer
    // CHECK:       [[BUF4:%.*]] = memref.alloc() : memref<1x64x16x16xf16, #NHWC, @DDR>

    // CHECK:       [[T0:%.*]] = async.execute
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs(%arg0 as [[ARG0:%.*]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[BUF0]] as [[ARG1:%.*]]: memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:                   VPUIP.Copy
    // CHECK-SAME:                  inputs([[ARG0]] : memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:                  outputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       [[T1:%.*]] = async.execute
    // CHECK-SAME:      [[T0]]
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[CST_WEIGHTS_TABLE]] as [[ARG4:%.*]]: memref<64x1x1x4xsi32, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[BUF2]] as [[ARG5:%.*]]: memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK:                   VPUIP.Copy
    // CHECK-SAME:                  inputs([[ARG4]] : memref<64x1x1x4xsi32, #NHWC, @DDR>)
    // CHECK-SAME:                  outputs([[ARG5]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)

    // CHECK:       [[T2:%.*]] = async.execute
    // CHECK-SAME:      [[T0]]
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[CST_WEIGHTS]] as [[ARG2:%.*]]: memref<64x32x3x3xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[BUF1]] as [[ARG3:%.*]]: memref<64x32x3x3xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:                   VPUIP.Copy
    // CHECK-SAME:                  inputs([[ARG2]] : memref<64x32x3x3xf16, #NHWC, @DDR>)
    // CHECK-SAME:                  outputs([[ARG3]] : memref<64x32x3x3xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       [[T3:%.*]] = async.execute
    // CHECK-SAME:      [[T0]], [[T2]], [[T1]]
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[BUF0]] as [[ARG6:%.*]]: memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>,
    // CHECK-SAME:                 [[BUF1]] as [[ARG7:%.*]]: memref<64x32x3x3xf16, #NHWC, [@CMX_NN, 0]>,
    // CHECK-SAME:                 [[BUF2]] as [[ARG8:%.*]]: memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[BUF3]] as [[ARG9:%.*]]: memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:                   VPUIP.NCEClusterTask
    // CHECK-SAME:                  input([[ARG6]] : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:                  weights([[ARG7]] : memref<64x32x3x3xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:                  weight_table([[ARG8]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:                  outputs([[ARG9]] : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       [[T4:%.*]] = async.execute
    // CHECK-SAME:      [[T2]], [[T3]]
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[BUF3]] as [[ARG10:%.*]]: memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[BUF4]] as [[ARG11:%.*]]: memref<1x64x16x16xf16, #NHWC, @DDR>)
    // CHECK:                   VPUIP.Copy
    // CHECK-SAME:                  inputs([[ARG10]] : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:                  outputs([[ARG11]] : memref<1x64x16x16xf16, #NHWC, @DDR>)

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!BufDistributed = !VPUIP.DistributedBuffer<
    1x32x64x64xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WtDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!ActWinDistributed = !VPUIP.DistributedBuffer<
    1x1x1x32xui8, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!BufMemrefDDR = memref<1x32x64x64xf16, #NHWC, @DDR>
!BufMemrefCMX = memref<1x32x64x64xf16, #NHWC, [@CMX_NN, 0]>

!WtMemrefDDR = memref<32x1x1x4xsi32, @DDR>
!WtMemrefCMX = memref<32x1x1x4xsi32, [@CMX_NN, 0]>

!ActWinMemrefDDR = memref<1x1x1x32xui8, @DDR>
!ActWinMemrefCMX = memref<1x1x1x32xui8, [@CMX_NN, 0]>

// CHECK-LABEL: @SpillingWithClustering
module @SpillingWithClustering {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x32x64x64xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x32x64x64xf16>
    }

func.func @main(%input: !BufMemrefDDR) -> !BufMemrefDDR {
    %cst0 = const.Declare memref<1x32x64x64xf16> = dense<2.0> : tensor<1x32x64x64xf16>
    %cst1 = const.Declare !WtMemrefCMX = dense<1> : tensor<32x1x1x4xsi32>
    %cst2 = const.Declare !ActWinMemrefCMX = dense<1> : tensor<1x1x1x32xui8>

    %buf_in = VPURT.AllocDistributed -> !BufDistributed
    %buf_wt = VPURT.AllocDistributed -> !WtDistributed
    %buf_act_win = VPURT.AllocDistributed -> !ActWinDistributed
    %buf0 = VPURT.AllocDistributed -> !BufDistributed
    %buf1 = VPURT.AllocDistributed -> !BufDistributed
    %buf2 = VPURT.AllocDistributed -> !BufDistributed
    %buf3 = VPURT.AllocDistributed -> !BufDistributed
    %output = memref.alloc() : !BufMemrefDDR

    %t_in, %r_in = async.execute -> !async.value<!BufDistributed> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !BufMemrefDDR) outputs(%buf_in as %arg1: !BufMemrefCMX) -> !BufDistributed {
            %1 = VPUIP.Copy inputs(%arg0 : !BufMemrefDDR) outputs(%arg1 : !BufMemrefCMX) -> !BufMemrefCMX
        }
        async.yield %0: !BufDistributed
    }

    %t0, %r0 = async.execute -> !async.value<!BufDistributed> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%cst0 as %arg0: !BufMemrefDDR) outputs(%buf0 as %arg1: !BufMemrefCMX) -> !BufDistributed {
            %1 = VPUIP.Copy inputs(%arg0 : !BufMemrefDDR) outputs(%arg1 : !BufMemrefCMX) -> !BufMemrefCMX
        }
        async.yield %0: !BufDistributed
    }

    %t10, %r10 = async.execute -> !async.value<!WtDistributed> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 2 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%cst1 as %arg0: !WtMemrefDDR) outputs(%buf_wt as %arg1: !WtMemrefCMX) -> !WtDistributed {
            %1 = VPUIP.Copy inputs(%arg0 : !WtMemrefDDR) outputs(%arg1 : !WtMemrefCMX) -> !WtMemrefCMX
        }
        async.yield %0: !WtDistributed
    }

    %t11, %r11 = async.execute -> !async.value<!ActWinDistributed> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%cst2 as %arg0: !ActWinMemrefDDR) outputs(%buf_act_win as %arg1: !ActWinMemrefCMX) -> !ActWinDistributed {
            %1 = VPUIP.Copy inputs(%arg0 : !ActWinMemrefDDR) outputs(%arg1 : !ActWinMemrefCMX) -> !ActWinMemrefCMX
        }
        async.yield %0: !ActWinDistributed
    }

    %t3, %r3 = async.execute [%t_in, %t10, %t11] (%r_in as %async_arg0 : !async.value<!BufDistributed>,
                                                  %r10 as %async_arg1 : !async.value<!WtDistributed>,
                                                  %r11 as %async_arg2 : !async.value<!ActWinDistributed>)
                -> !async.value<!BufDistributed> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 4 : i64, "async-deps-index" = 5 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%async_arg0 as %arg0: !BufMemrefCMX, %async_arg1 as %arg1: !WtMemrefCMX, %async_arg2 as %arg2: !ActWinMemrefCMX) outputs(%buf1 as %arg3: !BufMemrefCMX) -> !BufDistributed {
            %1 = VPUIP.NCEClusterTask {
                    activation_window_channel_length = 27 : i64,
                    kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = "MAXPOOL"
                }
                input(%arg0 : !BufMemrefCMX)
                weight_table(%arg1 : !WtMemrefCMX)
                activation_window(%arg2 : !ActWinMemrefCMX)
                parent_input(%arg0 : !BufMemrefCMX)
                parent_output(%arg3 : !BufMemrefCMX)
                outputs(%arg3 : !BufMemrefCMX) -> !BufMemrefCMX
                variants :
                {
                    DPUTask { outEnd = [16, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
                }
                PPE : {
                }
        }
        async.yield %0: !BufDistributed
    }

    %t1, %r1 = async.execute [%t0, %t10, %t11, %t3] (%r0 as %async_arg0 : !async.value<!BufDistributed>, %r3 as %async_arg1 : !async.value<!BufDistributed>)
            -> !async.value<!BufDistributed> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 4 : i64, "async-deps-index" = 6 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%async_arg0 as %arg0: !BufMemrefCMX, %async_arg1 as %arg1: !BufMemrefCMX) outputs(%buf2 as %arg2: !BufMemrefCMX) -> !BufDistributed {
            %1 = VPUIP.NCEClusterTask {
                    activation_window_channel_length = 0 : i64,
                    task_type = "ELTWISE"
                }
                input(%arg0 : !BufMemrefCMX)
                weights(%arg1 : !BufMemrefCMX)
                parent_input(%arg0 : !BufMemrefCMX)
                parent_output(%arg2 : !BufMemrefCMX)
                outputs(%arg2 : !BufMemrefCMX) -> !BufMemrefCMX
                variants :
                {
                    DPUTask { outEnd = [16, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
                }
                PPE : {
                    PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
                }
        }
        async.yield %0: !BufDistributed
    }

    %t5, %r5 = async.execute [%t_in, %t1] (%r_in as %async_arg0 : !async.value<!BufDistributed>, %r1 as %async_arg1 : !async.value<!BufDistributed>)
            -> !async.value<!BufDistributed> attributes {VPUIP.executor = @NCE, VPUIP.num_units = 4 : i64, "async-deps-index" = 7 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%async_arg0 as %arg0: !BufMemrefCMX, %async_arg1 as %arg1: !BufMemrefCMX) outputs(%buf3 as %arg2: !BufMemrefCMX) -> !BufDistributed {
            %1 = VPUIP.NCEClusterTask {
                    activation_window_channel_length = 0 : i64,
                    task_type = "ELTWISE"
                }
                input(%arg0 : !BufMemrefCMX)
                weights(%arg1 : !BufMemrefCMX)
                parent_input(%arg0 : !BufMemrefCMX)
                parent_output(%arg2 : !BufMemrefCMX)
                outputs(%arg2 : !BufMemrefCMX) -> !BufMemrefCMX
                variants :
                {
                    DPUTask { outEnd = [16, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
                }
                PPE : {
                    PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
                }
        }
        async.yield %0: !BufDistributed
    }

    %t6, %r6 = async.execute [%t5] (%r5 as %async_arg0 : !async.value<!BufDistributed>)
            -> !async.value<!BufMemrefDDR> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 8 : i64} {
        %0 = VPUIP.NCEClusterTiling inputs(%async_arg0 as %arg0: !BufMemrefCMX) outputs(%output as %arg1: !BufMemrefDDR) -> !BufMemrefDDR {
            %1 = VPUIP.Copy { out_mem_space = @DDR } inputs(%arg0: !BufMemrefCMX) outputs(%arg1: !BufMemrefDDR) -> !BufMemrefDDR
        }
        async.yield %0: !BufMemrefDDR
    }

     %6 = async.await %r6 : !async.value<!BufMemrefDDR>
     return %6 : !BufMemrefDDR

    // CHECK-DAG:       [[CST0:%.*]] = const.Declare memref<1x32x64x64xf16>
    // CHECK-DAG:       [[CST1:%.*]] = const.Declare memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK-DAG:       [[CST2:%.*]] = const.Declare memref<1x1x1x32xui8, [@CMX_NN, 0]>

    // CHECK:       [[BUF0:%.*]] = VPURT.DeclareBuffer "CMX_NN" <576> -> !VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUF1:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUF2:%.*]] = VPURT.DeclareBuffer "CMX_NN" <512> -> !VPUIP.DistributedBuffer<1x1x1x32xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUF3:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUF4:%.*]] = VPURT.DeclareBuffer "CMX_NN" <524864> -> !VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUF5:%.*]] = VPURT.DeclareBuffer "CMX_NN" <262144> -> !VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUF6:%.*]] = VPURT.DeclareBuffer "CMX_NN" <524288> -> !VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUF7:%.*]] = memref.alloc() : memref<1x32x64x64xf16, #NHWC, @DDR>
    // CHECK:       [[BUF_SPILL_WRITE:%.*]] = memref.alloc() : memref<1x32x64x64xf16, #NHWC, @DDR>
    // CHECK:       [[BUF_SPILL_READ:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[T0:%.*]], [[R0:%.*]] = async.execute
    // CHECK:         VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as [[ARG1:.*]]: memref<1x32x64x64xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BUF0]] as [[ARG2:.*]]: memref<1x32x64x64xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:             VPUIP.Copy
    // CHECK-SAME:          inputs([[ARG1]] : memref<1x32x64x64xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[ARG2]] : memref<1x32x64x64xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       [[T1:%.*]], [[R1:%.*]] = async.execute
    // CHECK-SAME:      [[T0]]
    // CHECK:         VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[CST1]] as [[ARG1:.*]]: memref<32x1x1x4xsi32, @DDR>)
    // CHECK-SAME:      outputs([[BUF1]] as [[ARG2:.*]]: memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK:             VPUIP.Copy
    // CHECK-SAME:          inputs([[ARG1]] : memref<32x1x1x4xsi32, @DDR>)
    // CHECK-SAME:          outputs([[ARG2]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)

    // CHECK:       [[T2:%.*]], [[R2:%.*]] = async.execute
    // CHECK-SAME:      [[T1]]
    // CHECK:         VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[CST2]] as [[ARG1:.*]]: memref<1x1x1x32xui8, @DDR>)
    // CHECK-SAME:      outputs([[BUF2]] as [[ARG2:.*]]: memref<1x1x1x32xui8, [@CMX_NN, 0]>)
    // CHECK:             VPUIP.Copy
    // CHECK-SAME:          inputs([[ARG1]] : memref<1x1x1x32xui8, @DDR>)
    // CHECK-SAME:          outputs([[ARG2]] : memref<1x1x1x32xui8, [@CMX_NN, 0]>)

    // CHECK:       [[T3:%.*]], [[R3:%.*]] = async.execute
    // CHECK-SAME:    [[T0]], [[T1]], [[T2]]
    // CHECK-SAME:    ([[R0]] as [[ARG1:%.*]]: !async.value<!VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK-SAME:     [[R1]] as [[ARG2:%.*]]: !async.value<!VPUIP.DistributedBuffer<32x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK-SAME:     [[R2]] as [[ARG3:%.*]]: !async.value<!VPUIP.DistributedBuffer<1x1x1x32xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>)
    // CHECK:            VPUIP.NCEClusterTiling
    // CHECK-SAME:         inputs([[ARG1]]
    // CHECK-SAME:                [[ARG2]]
    // CHECK-SAME:                [[ARG3]]
    // CHECK-SAME:         outputs([[BUF4]]
    // CHECK:                VPUIP.NCEClusterTask
    // CHECK-SAME:             task_type = "MAXPOOL"

    // CHECK:       [[T4:%.*]], [[R4:%.*]] = async.execute
    // CHECK-SAME:    [[T0]]
    // CHECK:            VPUIP.NCEClusterTiling
    // CHECK-SAME:         inputs([[BUF0]] as [[ARG1]]: memref<1x32x64x64xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:         outputs([[BUF_SPILL_WRITE]] as [[ARG2]]: memref<1x32x64x64xf16, #NHWC, @DDR>)
    // CHECK:                VPUIP.Copy
    // CHECK-SAME:             inputs([[ARG1]] : memref<1x32x64x64xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:             outputs([[ARG2]] : memref<1x32x64x64xf16, #NHWC, @DDR>)

    // CHECK:       [[T5:%.*]], [[R5:%.*]] = async.execute
    // CHECK-SAME:    [[T3]], [[T4]]
    // CHECK:         VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[CST0]] as [[ARG1:.*]]: memref<1x32x64x64xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BUF3]] as [[ARG2:.*]]: memref<1x32x64x64xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:             VPUIP.Copy
    // CHECK-SAME:          inputs([[ARG1]] : memref<1x32x64x64xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[ARG2]] : memref<1x32x64x64xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       [[T6:%.*]], [[R6:%.*]] = async.execute
    // CHECK-SAME:    [[T5]], [[T1]], [[T2]], [[T3]]
    // CHECK-SAME:    ([[R5]] as [[ARG1]]: !async.value<!VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK-SAME:     [[R3]] as [[ARG2]]: !async.value<!VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>)
    // CHECK-SAME:      -> !async.value<!VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[ARG1]]
    // CHECK-SAME:               [[ARG2]]
    // CHECK-SAME:        outputs([[BUF5]]
    // CHECK:               VPUIP.NCEClusterTask
    // CHECK-SAME:            task_type = "ELTWISE"

    // CHECK:       [[T7:%.*]], [[R7:%.*]] = async.execute
    // CHECK-SAME:    [[T5]], [[T6]], [[T4]]
    // CHECK-SAME:    ([[R4]] as [[ARG1]]: !async.value<memref<1x32x64x64xf16, #NHWC, @DDR>>)
    // CHECK:           VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[ARG1]] as [[ARG2]]: memref<1x32x64x64xf16, #NHWC, @DDR>)
    // CHECK-SAME:        outputs([[BUF_SPILL_READ]] as [[ARG3:%.*]]: memref<1x32x64x64xf16, #NHWC, @CMX_NN>)
    // CHECK:               VPUIP.Copy
    // CHECK-SAME:            inputs([[ARG2]] : memref<1x32x64x64xf16, #NHWC, @DDR>)
    // CHECK-SAME:            outputs([[ARG3]] : memref<1x32x64x64xf16, #NHWC, @CMX_NN>)

    // CHECK:       [[T8:%.*]], [[R8:%.*]] = async.execute
    // CHECK-SAME:    [[T0]], [[T6]], [[T7]]
    // CHECK-SAME:    ([[R7]] as [[ARG1]]: !async.value<!VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK-SAME:     [[R6]] as [[ARG2]]: !async.value<!VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>)
    // CHECK:         VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[ARG1]]
    // CHECK-SAME:             [[ARG2]]
    // CHECK-SAME:      outputs([[BUF6]]
    // CHECK:             VPUIP.NCEClusterTask
    // CHECK-SAME:          task_type = "ELTWISE"

    // CHECK:       [[T9:%.*]], [[R9:%.*]] = async.execute
    // CHECK-SAME:    [[T8]]
    // CHECK-SAME:    ([[R8]] as [[ARG1]]: !async.value<!VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>)
    // CHECK:         VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[ARG1]] as [[ARG2]]: memref<1x32x64x64xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[BUF7]] as [[ARG3]]: memref<1x32x64x64xf16, #NHWC, @DDR>)
    // CHECK:             VPUIP.Copy
    // CHECK-SAME:          inputs([[ARG2]] : memref<1x32x64x64xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[ARG3]] : memref<1x32x64x64xf16, #NHWC, @DDR>)
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Prefetching
module @Prefetching {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x32x16x16xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x128x4x4xf16>
    }

func.func @main(%in: memref<1x32x16x16xf16>, %out: memref<1x128x4x4xf16>) -> memref<1x128x4x4xf16> {

    %cst_4 = const.Declare memref<64x1x1x4xsi32> = dense<2> : tensor<64x1x1x4xsi32>
    %cst_10 = const.Declare memref<64x32x3x3xf16, #NHWC> = dense<2.0> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_3 = const.Declare memref<128x1x1x4xsi32> = dense<2> : tensor<128x1x1x4xsi32>
    %cst_11 = const.Declare memref<128x64x3x3xf16, #NHWC> = dense<2.0> : tensor<128x64x3x3xf16>, [#const.Reorder<#NHWC>]

    %buf0 = memref.alloc() : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %9 = memref.alloc() : memref<64x32x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %10 = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %11 = memref.alloc() : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>

    %12 = memref.alloc() : memref<128x64x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %13 = memref.alloc() : memref<128x1x1x4xsi32, [@CMX_NN, 0]>
    %14 = memref.alloc() : memref<1x128x4x4xf16, #NHWC, [@CMX_NN, 0]>

    %token_30, %results_31 = async.execute -> !async.value<memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, "cycleCost" = 683 : i64} {
      %32 = VPUIP.Copy inputs(%in : memref<1x32x16x16xf16>) outputs(%buf0 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
      async.yield %32 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
    }

    %token_32, %results_33 = async.execute -> !async.value<memref<64x32x3x3xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, "cycleCost" = 2172 : i64} {
      %32 = VPUIP.Copy inputs(%cst_10 : memref<64x32x3x3xf16, #NHWC>) outputs(%9 : memref<64x32x3x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x32x3x3xf16, #NHWC, [@CMX_NN, 0]>
      async.yield %32 : memref<64x32x3x3xf16, #NHWC, [@CMX_NN, 0]>
    }
    %token_34, %results_35 = async.execute -> !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, "cycleCost" = 22 : i64} {
      %32 = VPUIP.Copy inputs(%cst_4 : memref<64x1x1x4xsi32>) outputs(%10 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
      async.yield %32 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    }
    %token_36, %results_37 = async.execute [%token_30, %token_32, %token_34] (%results_31 as %arg2:
        !async.value<memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>>, %results_33 as %arg3:
        !async.value<memref<64x32x3x3xf16, #NHWC, [@CMX_NN, 0]>>, %results_35 as %arg4:
        !async.value<memref<64x1x1x4xsi32, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, "async-deps-index" = 3 : i64, "cycleCost" = 734 : i64} {
      %32 = VPUIP.NCEClusterTask
            {kernel_padding = {bottom = 1 : i64, left = 0 : i64, right = 1 : i64, top = 0 : i64},
            kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 734 : i64, task_type = "CONV"}
            input(%arg2 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%arg3 : memref<64x32x3x3xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%arg4 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%arg2 : memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%11 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%11 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>)
                -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]> variants :  {
        DPUTask {outEnd = [3, 3, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
        DPUTask {outEnd = [7, 3, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 0 : i64, right = 1 : i64, top = 0 : i64}, outStart = [4, 0, 0]}
        DPUTask {outEnd = [3, 7, 63], mpe_mode = "MATRIX", pad = {bottom = 1 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 4, 0]}
        DPUTask {outEnd = [7, 7, 63], mpe_mode = "MATRIX", pad = {bottom = 1 : i64, left = 0 : i64, right = 1 : i64, top = 0 : i64}, outStart = [4, 4, 0]}
      } PPE :  {
      }
      async.yield %32 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>
    }
    %token_38, %results_39 = async.execute -> !async.value<memref<128x64x3x3xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 4 : i64, "cycleCost" = 6144 : i64} {
      %32 = VPUIP.Copy inputs(%cst_11 : memref<128x64x3x3xf16, #NHWC>) outputs(%12 : memref<128x64x3x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<128x64x3x3xf16, #NHWC, [@CMX_NN, 0]>
      async.yield %32 : memref<128x64x3x3xf16, #NHWC, [@CMX_NN, 0]>
    }
    %token_40, %results_41 = async.execute -> !async.value<memref<128x1x1x4xsi32, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 5 : i64, "cycleCost" = 61 : i64} {
      %32 = VPUIP.Copy inputs(%cst_3 : memref<128x1x1x4xsi32>) outputs(%13 : memref<128x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<128x1x1x4xsi32, [@CMX_NN, 0]>
      async.yield %32 : memref<128x1x1x4xsi32, [@CMX_NN, 0]>
    }
    %token_42, %results_43 = async.execute [%token_36, %token_38, %token_40] (%results_37 as %arg2:
        !async.value<memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>>, %results_39 as %arg3:
        !async.value<memref<128x64x3x3xf16, #NHWC, [@CMX_NN, 0]>>, %results_41 as %arg4:
        !async.value<memref<128x1x1x4xsi32, [@CMX_NN, 0]>>) ->
        !async.value<memref<1x128x4x4xf16, #NHWC, [@CMX_NN, 0]>> attributes {VPUIP.executor = @NCE, "async-deps-index" = 6 : i64, "cycleCost" = 686 : i64} {
      %32 = VPUIP.NCEClusterTask
            {kernel_padding = {bottom = 1 : i64, left = 0 : i64, right = 1 : i64, top = 0 : i64}, kernel_size = [3, 3],
            kernel_strides = [2, 2], minimumHardwareExecutionCost = 686 : i64, task_type = "CONV"}
            input(%arg2 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%arg3 : memref<128x64x3x3xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%arg4 : memref<128x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%arg2 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%14 : memref<1x128x4x4xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%14 : memref<1x128x4x4xf16, #NHWC, [@CMX_NN, 0]>)
                -> memref<1x128x4x4xf16, #NHWC, [@CMX_NN, 0]> variants :  {
        DPUTask {outEnd = [3, 3, 31], mpe_mode = "MATRIX", pad = {bottom = 1 : i64, left = 0 : i64, right = 1 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
        DPUTask {outEnd = [3, 3, 63], mpe_mode = "MATRIX", pad = {bottom = 1 : i64, left = 0 : i64, right = 1 : i64, top = 0 : i64}, outStart = [0, 0, 32]}
        DPUTask {outEnd = [3, 3, 95], mpe_mode = "MATRIX", pad = {bottom = 1 : i64, left = 0 : i64, right = 1 : i64, top = 0 : i64}, outStart = [0, 0, 64]}
        DPUTask {outEnd = [3, 3, 127], mpe_mode = "MATRIX", pad = {bottom = 1 : i64, left = 0 : i64, right = 1 : i64, top = 0 : i64}, outStart = [0, 0, 96]}
      } PPE :  {
      }
      async.yield %32 : memref<1x128x4x4xf16, #NHWC, [@CMX_NN, 0]>
    }

    %token_43, %result_44 = async.execute [%token_42] (%results_43 as %0 : !async.value<memref<1x128x4x4xf16, #NHWC, [@CMX_NN, 0]>>)
            -> !async.value<memref<1x128x4x4xf16>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 7 : i64, "cycleCost" = 171 : i64} {
        %1 = VPUIP.Copy inputs(%0 : memref<1x128x4x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x128x4x4xf16>) -> memref<1x128x4x4xf16>
        async.yield %1 : memref<1x128x4x4xf16>
    }

    %44 = async.await %result_44 : !async.value<memref<1x128x4x4xf16>>
    return %44 : memref<1x128x4x4xf16>

    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 683 : i64, cycleEnd = 683 : i64}
    // CHECK:           VPUIP.Copy
    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 2 : i64, cycleBegin = 683 : i64, cycleCost = 22 : i64, cycleEnd = 705 : i64}
    // CHECK:           VPUIP.Copy
    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 1 : i64, cycleBegin = 705 : i64, cycleCost = 2172 : i64, cycleEnd = 2877 : i64}
    // CHECK:           VPUIP.Copy

    // CHECK:       {VPUIP.executor = @NCE, "async-deps-index" = 3 : i64, cycleBegin = 2877 : i64, cycleCost = 734 : i64, cycleEnd = 3611 : i64}
    // CHECK:           VPUIP.NCEClusterTask

                    // Prefetched Copy executing in parallel cycles to NCEClusterTask
    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 4 : i64, cycleBegin = 2877 : i64, cycleCost = 6144 : i64, cycleEnd = 9021 : i64}
    // CHECK:           VPUIP.Copy
    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 5 : i64, cycleBegin = 9021 : i64, cycleCost = 61 : i64, cycleEnd = 9082 : i64}
    // CHECK:           VPUIP.Copy

    // CHECK:       {VPUIP.executor = @NCE, "async-deps-index" = 6 : i64, cycleBegin = 9082 : i64, cycleCost = 686 : i64, cycleEnd = 9768 : i64}
    // CHECK:           VPUIP.NCEClusterTask
    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 7 : i64, cycleBegin = 9768 : i64, cycleCost = 171 : i64, cycleEnd = 9939 : i64}
    // CHECK:           VPUIP.Copy
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NoPrefetching
module @NoPrefetching {

  IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x16x3xui8>
    }
    outputsInfo : {
        DataInfo "power" : tensor<1x16x16x3xf32>
    }

func.func @main(%arg0: memref<1x16x16x3xui8, @DDR>, %arg1: memref<1x16x16x3xf32, @DDR>) -> memref<1x16x16x3xf32, @DDR> {
    %cst = const.Declare memref<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %0 = memref.alloc() : memref<1x16x16x3xui8, [@CMX_NN, 0]>
    %1 = memref.alloc() : memref<1x3x16x16xui8, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x3x16x16xui8, @DDR>
    %3 = memref.alloc() : memref<1x3x16x16xf16, @DDR>
    %4 = memref.alloc() : memref<1x3x16x16xf16, @DDR>
    %5 = memref.alloc() : memref<1x16x16x3xf16, @DDR>

    %token, %results = async.execute -> !async.value<memref<1x16x16x3xui8, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleCost = 27 : i64} {
      %7 = VPUIP.Copy inputs(%arg0 : memref<1x16x16x3xui8, @DDR>) outputs(%0 : memref<1x16x16x3xui8, [@CMX_NN, 0]>) -> memref<1x16x16x3xui8, [@CMX_NN, 0]>
      async.yield %7 : memref<1x16x16x3xui8, [@CMX_NN, 0]>
    }
    %token_0, %results_1 = async.execute [%token] (%results as %arg2: !async.value<memref<1x16x16x3xui8, [@CMX_NN, 0]>>) -> !async.value<memref<1x3x16x16xui8, [@CMX_NN, 0]>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, cycleCost = 27 : i64} {
      %7 = VPUIP.PermuteDMA {mem_perm = #NWCH, port = 0 : i64} inputs(%arg2 : memref<1x16x16x3xui8, [@CMX_NN, 0]>) outputs(%1 : memref<1x3x16x16xui8, [@CMX_NN, 0]>) -> memref<1x3x16x16xui8, [@CMX_NN, 0]>
      async.yield %7 : memref<1x3x16x16xui8, [@CMX_NN, 0]>
    }
    %token_2, %results_3 = async.execute [%token_0] (%results_1 as %arg2: !async.value<memref<1x3x16x16xui8, [@CMX_NN, 0]>>) -> !async.value<memref<1x3x16x16xui8, @DDR>> attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleCost = 27 : i64} {
      %7 = VPUIP.Copy inputs(%arg2 : memref<1x3x16x16xui8, [@CMX_NN, 0]>) outputs(%2 : memref<1x3x16x16xui8, @DDR>) -> memref<1x3x16x16xui8, @DDR>
      async.yield %7 : memref<1x3x16x16xui8, @DDR>
    }
    %token_4, %results_5 = async.execute [%token_2] (%results_3 as %arg2: !async.value<memref<1x3x16x16xui8, @DDR>>) -> !async.value<memref<1x3x16x16xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 3 : i64, cycleCost = 2 : i64} {
      %7 = VPUIP.ConvertUPA inputs(%arg2 : memref<1x3x16x16xui8, @DDR>) outputs(%3 : memref<1x3x16x16xf16, @DDR>) -> memref<1x3x16x16xf16, @DDR>
      async.yield %7 : memref<1x3x16x16xf16, @DDR>
    }
    %token_6, %results_7 = async.execute [%token_4] (%results_5 as %arg2: !async.value<memref<1x3x16x16xf16, @DDR>>) -> !async.value<memref<1x3x16x16xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 4 : i64, cycleCost = 2 : i64} {
      %7 = VPUIP.EltwiseUPA {type = "POWER"} inputs(%arg2 : memref<1x3x16x16xf16, @DDR>, %cst : memref<1x1x1x1xf16>) outputs(%4 : memref<1x3x16x16xf16, @DDR>) -> memref<1x3x16x16xf16, @DDR>
      async.yield %7 : memref<1x3x16x16xf16, @DDR>
    }
    %token_8, %results_9 = async.execute [%token_6] (%results_7 as %arg2: !async.value<memref<1x3x16x16xf16, @DDR>>) -> !async.value<memref<1x16x16x3xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 5 : i64, cycleCost = 2 : i64} {
      %7 = VPUIP.PermuteUPA {order_value = #NHWC} inputs(%arg2 : memref<1x3x16x16xf16, @DDR>) outputs(%5 : memref<1x16x16x3xf16, @DDR>) -> memref<1x16x16x3xf16, @DDR>
      async.yield %7 : memref<1x16x16x3xf16, @DDR>
    }
    %token_10, %results_11 = async.execute [%token_8] (%results_9 as %arg2: !async.value<memref<1x16x16x3xf16, @DDR>>) -> !async.value<memref<1x16x16x3xf32, @DDR>> attributes {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 6 : i64, cycleCost = 2 : i64} {
      %7 = VPUIP.ConvertUPA inputs(%arg2 : memref<1x16x16x3xf16, @DDR>) outputs(%arg1 : memref<1x16x16x3xf32, @DDR>) -> memref<1x16x16x3xf32, @DDR>
      async.yield %7 : memref<1x16x16x3xf32, @DDR>
    }
    %6 = async.await %results_11 : !async.value<memref<1x16x16x3xf32, @DDR>>
    return %6 : memref<1x16x16x3xf32, @DDR>

    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 27 : i64, cycleEnd = 27 : i64}
    // CHECK:           VPUIP.Copy
    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 1 : i64, cycleBegin = 27 : i64, cycleCost = 27 : i64, cycleEnd = 54 : i64}
    // CHECK:           VPUIP.PermuteDMA
    // CHECK:       {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 2 : i64, cycleBegin = 54 : i64, cycleCost = 27 : i64, cycleEnd = 81 : i64}
    // CHECK:           VPUIP.Copy

    // CHECK:       {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 3 : i64, cycleBegin = 81 : i64, cycleCost = 2 : i64, cycleEnd = 83 : i64}
    // CHECK:           VPUIP.ConvertUPA
    // CHECK:       {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 4 : i64, cycleBegin = 83 : i64, cycleCost = 2 : i64, cycleEnd = 85 : i64}
    // CHECK:           VPUIP.EltwiseUPA
    // CHECK:       {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 5 : i64, cycleBegin = 85 : i64, cycleCost = 2 : i64, cycleEnd = 87 : i64}
    // CHECK:           VPUIP.PermuteUPA
    // CHECK:       {VPUIP.executor = @SHAVE_UPA, "async-deps-index" = 6 : i64, cycleBegin = 87 : i64, cycleCost = 2 : i64, cycleEnd = 89 : i64}
    // CHECK:           VPUIP.ConvertUPA
}

}

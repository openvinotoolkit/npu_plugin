//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-eltwise-to-in-place --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InplaceEltwiseSameType
func @InplaceEltwiseSameType(%in: memref<1x32x96x96xf16, #NHWC>, %out: memref<1x32x96x96xf16, #NHWC>) -> memref<1x32x96x96xf16, #NHWC> {
    %cst0 = const.Declare memref<1x32x96x96xf16, #NHWC> = dense<2.0> : tensor<1x32x96x96xf16>, [#const.Reorder<#NHWC>]

    %buf_in = memref.alloc() : memref<1x32x96x96xf16, #NHWC, @CMX_NN>
    %buf0 = memref.alloc() : memref<1x32x96x96xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x32x96x96xf16, #NHWC, @CMX_NN>

    %0 = VPUIP.Copy inputs(%in : memref<1x32x96x96xf16, #NHWC>) outputs(%buf_in : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) -> memref<1x32x96x96xf16, #NHWC, @CMX_NN>

    %1 = VPUIP.Copy inputs(%cst0 : memref<1x32x96x96xf16, #NHWC>) outputs(%buf0 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) -> memref<1x32x96x96xf16, #NHWC, @CMX_NN>

    %2 = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                task_type = "ELTWISE",
                is_inplace = true
            }
            input(%0 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
            weights(%1 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
            parent_input(%0 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
            parent_output(%buf1 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
            outputs(%buf1 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) -> memref<1x32x96x96xf16, #NHWC, @CMX_NN>
            variants :
            {
                DPUTask { outEnd = [32, 96, 96], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
            }
            PPE : {
                PPETask "ADD" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }

    %3 = VPUIP.Copy inputs(%2 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) outputs(%out : memref<1x32x96x96xf16, #NHWC>) -> memref<1x32x96x96xf16, #NHWC>

    return %3 : memref<1x32x96x96xf16, #NHWC>

    // CHECK:       [[BUF0:%.*]] = memref.alloc()
    // CHECK:       [[BUF1:%.*]] = memref.alloc()
    // CHECK-NOT:   [[BUF3:%.*]] = memref.alloc()

    // CHECK:       [[VAL0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
    // CHECK:       [[VAL1:%.*]] = VPUIP.Copy
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)

    // CHECK:       [[VAL2:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = "ELTWISE"
    // CHECK-SAME:      input([[VAL0]] : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) weights([[VAL1]] : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)


    // CHECK:       [[VAL3:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[VAL2]] : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)

    //CHECK:        return [[VAL3]] : memref<1x32x96x96xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InplaceEltwiseFpClusterTiling
// CHECK-SAME:    ([[ARG0:%.*]]: memref<1x256x56x56xf16, #NHWC>,
// CHECK-SAME:    [[ARG1:%.*]]: memref<1x256x56x56xf16, #NHWC>)
// CHECK-SAME:    -> memref<1x256x56x56xf16, #NHWC> {
func @InplaceEltwiseFpClusterTiling(%in: memref<1x256x56x56xf16, #NHWC>, %out: memref<1x256x56x56xf16, #NHWC>) -> memref<1x256x56x56xf16, #NHWC> {
    
    %cst0 = const.Declare memref<1x256x56x56xf16, #NHWC> = dense<2.0> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]

    %buf_0  = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %buf_in = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %output_buf = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    %0 = VPUIP.NCEClusterTiling 
        inputs(%cst0 as %arg2: memref<1x256x56x56xf16, #NHWC>) 
        outputs(%buf_0 as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            %internal_0 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN>
    }

    %1 = VPUIP.NCEClusterTiling
        inputs(%in as %arg2: memref<1x256x56x56xf16, #NHWC>) 
        outputs(%buf_in as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            %internal_1 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN>
    }

    %2 = VPUIP.NCEClusterTiling 
    inputs(
        %0 as %arg2: memref<1x256x56x56xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>, 
        %1 as %arg3: memref<1x256x56x56xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
    outputs(
        %output_buf as %arg4: memref<1x256x56x56xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) 
    -> !VPUIP.DistributedBuffer<1x256x56x56xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %internal_2 = VPUIP.NCEClusterTask 
            {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 31170 : i64, task_type = "ELTWISE"} 
            input(%arg2 : memref<1x256x56x56xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) 
            weights(%arg3 : memref<1x256x56x56xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) 
            parent_input(%arg2 : memref<1x256x56x56xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) 
            parent_output(%arg4 : memref<1x256x56x56xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) 
            outputs(%arg4 : memref<1x256x56x56xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>) 
                -> memref<1x256x56x56xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_8x16", outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_8x16", outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            } PPE : {
            PPETask "LRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}
            }
    }

    %3 = VPUIP.NCEClusterTiling
        inputs(%2 as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>) 
        outputs(%out as %arg3: memref<1x256x56x56xf16, #NHWC>)
        -> memref<1x256x56x56xf16, #NHWC> {
            %internal_3 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC>) -> memref<1x256x56x56xf16, #NHWC>
    }

    return %3 : memref<1x256x56x56xf16, #NHWC>

    //CHECK:        [[CST:%.*]] = const.Declare memref<1x256x56x56xf16, #NHWC> = dense<2.000000e+00> : tensor<1x256x56x56xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[BUF_IN0:%.*]] = VPURT.AllocDistributed
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:        [[BUF_IN1:%.*]] = VPURT.AllocDistributed
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-NOT:    VPURT.AllocDistributed

    //CHECK:        [[COPY_IN0:%.*]] = VPUIP.NCEClusterTiling 
    //CHECK-SAME:       inputs([[CST]] as %arg2: memref<1x256x56x56xf16, #NHWC>) 
    //CHECK-SAME:       outputs([[BUF_IN0]] as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            %6 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN>
    //CHECK:        }

    //CHECK:        [[COPY_IN1:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:       inputs([[ARG0]] as %arg2: memref<1x256x56x56xf16, #NHWC>)
    //CHECK-SAME:       outputs([[BUF_IN1]] as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            %6 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN>
    //CHECK:        }

    //CHECK:        [[ELTW_RES:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:       inputs([[COPY_IN0]] as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>, [[COPY_IN1]] as %arg3: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:       outputs([[BUF_IN0]] as %arg4: memref<1x256x56x56xf16, #NHWC, @CMX_NN>) 
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x256x56x56xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            %6 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 31170 : i64, task_type = "ELTWISE"} input(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) parent_input(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) parent_output(%arg4 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x256x56x56xf16, #NHWC, @CMX_NN> variants : {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_8x16", outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_8x16", outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:            } PPE : {
    //CHECK:                PPETask "LRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}
    //CHECK:            }
    //CHECK:        }

    //CHECK:        [[OUT_COPY:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:        inputs([[ELTW_RES]] as %arg2: memref<1x256x56x56xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:        outputs([[ARG1]] as %arg3: memref<1x256x56x56xf16, #NHWC>) -> memref<1x256x56x56xf16, #NHWC> {
    //CHECK:            %6 = VPUIP.Copy inputs(%arg2 : memref<1x256x56x56xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x256x56x56xf16, #NHWC>) -> memref<1x256x56x56xf16, #NHWC>
    //CHECK:        }

    //CHECK:        return [[OUT_COPY]] : memref<1x256x56x56xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.011588541666666667:128>
!qElemType1 = type !quant.uniform<u8:f16, 0.020557598039215686:128>
!qElemType2 = type !quant.uniform<u8:f16, 0.0088848039215686271>

!qType0 = type memref<1x256x56x56x!qElemType0, #NHWC>
!qType1 = type memref<1x256x56x56x!qElemType1, #NHWC>
!qType2 = type memref<1x256x56x56x!qElemType2, #NHWC>

!qType0CMX = type memref<1x256x56x56x!qElemType0, #NHWC, @CMX_NN>
!qType1CMX = type memref<1x256x56x56x!qElemType1, #NHWC, @CMX_NN>
!qType2CMX = type memref<1x256x56x56x!qElemType2, #NHWC, @CMX_NN>

// CHECK-LABEL: @InplaceEltwiseQuantizedView
// CHECK-SAME:    ([[ARG0:%.*]]: memref<1x256x56x56x!qElemType0, #NHWC>,
// CHECK-SAME:    [[ARG1:%.*]]: memref<1x256x56x56x!qElemType1, #NHWC>,
// CHECK-SAME:    [[ARG2:%.*]]: memref<1x256x56x56x!qElemType2, #NHWC>)
// CHECK-SAME:    -> memref<1x256x56x56x!qElemType2, #NHWC> {

func @InplaceEltwiseQuantizedView(%in: !qType0, %in2: !qType1, %out: !qType2) -> !qType2 {
    %buf_in = memref.alloc() : !qType0CMX
    %buf0 = memref.alloc() : !qType1CMX
    %buf1 = memref.alloc() : !qType2CMX

    %0 = VPUIP.Copy inputs(%in : !qType0) outputs(%buf_in : !qType0CMX) -> !qType0CMX
    %1 = VPUIP.Copy inputs(%in2 : !qType1) outputs(%buf0 : !qType1CMX) -> !qType1CMX

    %2 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 11669 : i64, task_type = "ELTWISE"} 
        input(%0 : !qType0CMX)
        weights(%1 : !qType1CMX)
        parent_input(%0 : !qType0CMX)
        parent_output(%buf1 : !qType2CMX)
        outputs(%buf1 : !qType2CMX) 
        -> !qType2CMX variants : {
                DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_8x16", outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
                DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_8x16", outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        } PPE : {
                PPETask "LRELUX" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [24302], in2_quant_mult = [43112], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [28813], quant_post_shift = 0 : i64, quant_shift = [29]}
        }

    %3 = VPUIP.Copy inputs(%2 : !qType2CMX) outputs(%out : !qType2) -> !qType2

    return %3 : !qType2

    // CHECK:       [[BUF0:%.*]] = memref.alloc() : memref<1x256x56x56x!qElemType0, #NHWC, @CMX_NN>
    // CHECK:       [[VIEW0:%.*]] = VPUIP.ViewOp [[BUF0]] : memref<1x256x56x56x!qElemType0, #NHWC, @CMX_NN> to memref<1x256x56x56x!qElemType2, #NHWC, @CMX_NN>
    // CHECK:       [[BUF2:%.*]] = memref.alloc() : memref<1x256x56x56x!qElemType1, #NHWC, @CMX_NN>

    // CHECK:       [[INP1:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x256x56x56x!qElemType0, #NHWC>) outputs([[BUF0]] : memref<1x256x56x56x!qElemType0, #NHWC, @CMX_NN>) -> memref<1x256x56x56x!qElemType0, #NHWC, @CMX_NN>
    // CHECK:       [[INP2:%.*]] = VPUIP.Copy inputs([[ARG1]] : memref<1x256x56x56x!qElemType1, #NHWC>) outputs([[BUF2]] : memref<1x256x56x56x!qElemType1, #NHWC, @CMX_NN>) -> memref<1x256x56x56x!qElemType1, #NHWC, @CMX_NN>
    // CHECK:       [[ELTWISE_OUT:%.*]] = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 11669 : i64, task_type = "ELTWISE"} 
    // CHECK-SAME:      input([[INP1]] : memref<1x256x56x56x!qElemType0, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights([[INP2]] : memref<1x256x56x56x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_input([[INP1]] : memref<1x256x56x56x!qElemType0, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[VIEW0]] : memref<1x256x56x56x!qElemType2, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[VIEW0]] : memref<1x256x56x56x!qElemType2, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> memref<1x256x56x56x!qElemType2, #NHWC, @CMX_NN> variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_8x16", outEnd = [55, 27, 255], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    // CHECK:           DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_8x16", outEnd = [55, 55, 255], outStart = [0, 28, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    // CHECK:          } PPE : {
    // CHECK:           PPETask "LRELUX" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [24302], in2_quant_mult = [43112], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [28813], quant_post_shift = 0 : i64, quant_shift = [29]}
    // CHECK:           }

    // CHECK:       [[COPY_OUT:%.*]] = VPUIP.Copy inputs([[ELTWISE_OUT]] : memref<1x256x56x56x!qElemType2, #NHWC, @CMX_NN>) 
    // CHECK-SAME:      outputs([[ARG2]] : memref<1x256x56x56x!qElemType2, #NHWC>) -> memref<1x256x56x56x!qElemType2, #NHWC>

    // CHECK:    return [[COPY_OUT]] : memref<1x256x56x56x!qElemType2, #NHWC>
}

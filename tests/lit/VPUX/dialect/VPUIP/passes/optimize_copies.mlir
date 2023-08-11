//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-copies %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OptimizeCopy(
        %arg0: memref<1x16x112x112xf16, #NHWC, @CMX_NN>,
        %arg1: memref<1x16x112x112xf16, #NHWC, @CMX_NN>,
        %arg2: memref<1x32x112x112xf16, #NHWC>)
        -> memref<1x32x112x112xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x32x112x112xf16, #NHWC>
    %1 = memref.alloc() : memref<1x16x112x112xf16, #NHWC>

    %2 = VPUIP.Copy inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%1 : memref<1x16x112x112xf16, #NHWC>)
        -> memref<1x16x112x112xf16, #NHWC>

    %3 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 112, 112] :
        memref<1x32x112x112xf16, #NHWC> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    %4 = VPUIP.Copy inputs(%2 : memref<1x16x112x112xf16, #NHWC>)
        outputs(%3 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>)
        -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>

    %5 = memref.alloc() : memref<1x16x112x112xf16, #NHWC>

    %6 = VPUIP.Copy inputs(%arg1 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%5 : memref<1x16x112x112xf16, #NHWC>)
        -> memref<1x16x112x112xf16, #NHWC>

    %7 = VPUIP.SubView %0 [0, 16, 0, 0] [1, 16, 112, 112] :
        memref<1x32x112x112xf16, #NHWC> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    %8 = VPUIP.Copy inputs(%6 : memref<1x16x112x112xf16, #NHWC>)
        outputs(%7 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>)
        -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>

    %9 = VPUIP.ConcatView
        inputs(%4, %8 :
            memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>,
            memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
        )
        outputs(%0 : memref<1x32x112x112xf16, #NHWC>)
        -> memref<1x32x112x112xf16, #NHWC>

    %10 = VPUIP.Copy inputs(%9 : memref<1x32x112x112xf16, #NHWC>)
        outputs(%arg2 : memref<1x32x112x112xf16, #NHWC>)
        -> memref<1x32x112x112xf16, #NHWC>

    return %10 : memref<1x32x112x112xf16, #NHWC>

    // CHECK-NOT:   memref.alloc() : memref<1x32x112x112xf16, #NHWC>

    // CHECK-NOT:   memref.alloc() : memref<1x16x112x112xf16, #NHWC>
    // CHECK:       [[VAR0:%.*]] = VPUIP.SubView %arg2 [0, 0, 0, 0] [1, 16, 112, 112] :
    // CHECK-SAME:      memref<1x32x112x112xf16, #NHWC> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    // CHECK:       [[VAR1:%.*]] = VPUIP.Copy inputs({{.*}} : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>)

    // CHECK-NOT:   memref.alloc() : memref<1x16x112x112xf16, #NHWC>
    // CHECK:       [[VAR2:%.*]] = VPUIP.SubView %arg2 [0, 16, 0, 0] [1, 16, 112, 112] :
    // CHECK-SAME:      memref<1x32x112x112xf16, #NHWC> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    // CHECK:       [[VAR3:%.*]] = VPUIP.Copy inputs({{.*}} : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>)

    // CHECK:       [[VAR4:%.*]] = VPUIP.ConcatView inputs([[VAR1]], [[VAR3]] :
    // CHECK-SAME:      outputs(%arg2 : memref<1x32x112x112xf16, #NHWC>)

    // CHECK: return [[VAR4]] : memref<1x32x112x112xf16, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @OptimizeLastCopyForPureViewOps(%arg0: memref<1x16x2x2xf16>, %arg1: memref<1x16x2x2xf16>, %arg2: memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16> {
    %0 = memref.alloc() : memref<1x64x2x2xf16>

    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %2 = VPUIP.Copy inputs(%arg0 : memref<1x16x2x2xf16>) outputs(%1 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %3 = VPUIP.SubView %0 [0, 16, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %4 = VPUIP.Copy inputs(%arg1 : memref<1x16x2x2xf16>) outputs(%3 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>

    %5 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 32, 2, 2] : memref<1x64x2x2xf16> to memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>
    %6 = VPUIP.ConcatView inputs(%2, %4 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>, memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) outputs(%5 : memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>) -> memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>

    %7 = VPUIP.SubView %0 [0, 32, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %8 = VPUIP.Copy inputs(%arg0 : memref<1x16x2x2xf16>) outputs(%7 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %9 = VPUIP.SubView %0 [0, 48, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    %10 = VPUIP.Copy inputs(%arg1 : memref<1x16x2x2xf16>) outputs(%9 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>

    %11 = VPUIP.SubView %0 [0, 32, 0, 0] [1, 32, 2, 2] : memref<1x64x2x2xf16> to
            memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>
    %12 = VPUIP.ConcatView inputs(%8, %10 : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>, memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)
            outputs(%11 : memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>) -> memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>

    %13 = VPUIP.ConcatView inputs(%6, %12 : memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>, memref<1x32x2x2xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [256, 4, 2, 1]}>)
            outputs(%0 : memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16>
    %14 = VPUIP.Copy inputs(%13 : memref<1x64x2x2xf16>) outputs(%arg2 : memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16>

    return %14 : memref<1x64x2x2xf16>

    // CHECK-NOT: memref.alloc() : memref<1x64x2x2xf16>

    // CHECK: [[VAR0:%.*]] = VPUIP.SubView %arg2 [0, 0, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK: [[VAR1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x2x2xf16>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)
    // CHECK: [[VAR2:%.*]] = VPUIP.SubView %arg2 [0, 16, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK: [[VAR3:%.*]] = VPUIP.Copy inputs(%arg1 : memref<1x16x2x2xf16>)
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)

    // CHECK: [[VAR4:%.*]] = VPUIP.SubView %arg2 [0, 0, 0, 0] [1, 32, 2, 2] : memref<1x64x2x2xf16> to memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK: [[VAR5:%.*]] = VPUIP.ConcatView inputs([[VAR1]], [[VAR3]] :
    // CHECK-SAME:      outputs([[VAR4]] : memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) -> memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>

    // CHECK: [[VAR6:%.*]] = VPUIP.SubView %arg2 [0, 32, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK: [[VAR7:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x2x2xf16>)
    // CHECK-SAME:      outputs([[VAR6]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)
    // CHECK: [[VAR8:%.*]] = VPUIP.SubView %arg2 [0, 48, 0, 0] [1, 16, 2, 2] : memref<1x64x2x2xf16> to memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK: [[VAR9:%.*]] = VPUIP.Copy inputs(%arg1 : memref<1x16x2x2xf16>)
    // CHECK-SAME:      outputs([[VAR8]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)

    // CHECK: [[VAR10:%.*]] = VPUIP.SubView %arg2 [0, 32, 0, 0] [1, 32, 2, 2] : memref<1x64x2x2xf16> to memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>
    // CHECK: [[VAR11:%.*]] = VPUIP.ConcatView inputs([[VAR7]], [[VAR9]] : memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>, memref<1x16x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)
    // CHECK-SAME:      outputs([[VAR10]] : memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>)

    // CHECK: [[VAR12:%.*]] = VPUIP.ConcatView inputs([[VAR5]], [[VAR11]] : memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>, memref<1x32x2x2xf16, {order = #NCHW, strides = [256, 4, 2, 1]}>) outputs(%arg2 : memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16>
    // CHECK-NOT: VPUIP.Copy

    // CHECK: return [[VAR12]] : memref<1x64x2x2xf16>
}

// -----

func.func @NoChangesSourceIsConstantOp(%arg0: memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16> {
    %0 = const.Declare memref<1x2x4x4xf16> = dense<1.000000e+00> : tensor<1x2x4x4xf16>
    %1 = VPUIP.Copy inputs(%0 : memref<1x2x4x4xf16>) outputs(%arg0 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>
    return %1 : memref<1x2x4x4xf16>

    // CHECK-DAG: [[VAR0:%.*]] = const.Declare
    // CHECK: [[VAR1:%.*]] = VPUIP.Copy
    // CHECK: return [[VAR1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NoChangesDifferentMemSpace(%arg0: memref<1x16x4x4xf16, #NHWC>, %arg1 : memref<16x1x1x4xsi32, @CMX_NN>,
                                 %arg2 : memref<16x1x1x16xui8, @CMX_NN>, %arg3: memref<1x16x2x2xf16, #NHWC>) -> memref<1x16x2x2xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, @CMX_NN>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x4xf16, #NHWC>) outputs(%0 : memref<1x16x4x4xf16, #NHWC, @CMX_NN>) -> memref<1x16x4x4xf16, #NHWC, @CMX_NN>

    %2 = memref.alloc() : memref<1x16x2x2xf16, #NHWC, @CMX_NN>
    %3 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [2, 2],
            kernel_strides = [2, 2],
            task_type = "MAXPOOL"
        }
        input(%1 : memref<1x16x4x4xf16, #NHWC, @CMX_NN>)
        weight_table(%arg1 : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%arg2 : memref<16x1x1x16xui8, @CMX_NN>)
        parent_input(%1 : memref<1x16x4x4xf16, #NHWC, @CMX_NN>)
        parent_output(%2 : memref<1x16x2x2xf16, #NHWC, @CMX_NN>)
        outputs(%2 : memref<1x16x2x2xf16, #NHWC, @CMX_NN>) -> memref<1x16x2x2xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 2, 2], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0] }
        }
        PPE : {
        }

    %4 = VPUIP.Copy inputs(%3 : memref<1x16x2x2xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x16x2x2xf16, #NHWC>) -> memref<1x16x2x2xf16, #NHWC>
    return %4 : memref<1x16x2x2xf16, #NHWC>

    // CHECK: VPUIP.Copy

    // CHECK: [[VAR0:%.*]] = VPUIP.NCEClusterTask
    // CHECK: [[VAR1:%.*]] = VPUIP.Copy inputs([[VAR0]] : memref<1x16x2x2xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                      outputs(%arg3 : memref<1x16x2x2xf16, #NHWC>)

    // CHECK: return [[VAR1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @CopiesWithSubViewOps(%act : memref<1x80x28x28xf16, #NHWC, @DDR>)
                              -> memref<1x80x28x27xf16, #NHWC, @DDR>
{
    %buf0 = memref.alloc() : memref<1x70x28x27xf16, #NHWC, @DDR>
    %buf1 = memref.alloc() : memref<1x80x28x27xf16, #NHWC, @DDR>
    %0 = VPUIP.SubView %act [0, 0, 0, 1] [1, 70, 28, 27] : memref<1x80x28x28xf16, #NHWC, @DDR> to memref<1x70x28x27xf16, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>
    %1 = VPUIP.Copy inputs(%0 : memref<1x70x28x27xf16, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>) outputs(%buf0 : memref<1x70x28x27xf16, #NHWC, @DDR>) -> memref<1x70x28x27xf16, #NHWC, @DDR>
    %2 = VPUIP.SubView %buf1 [0, 0, 0, 0] [1, 70, 28, 27] : memref<1x80x28x27xf16, #NHWC, @DDR> to memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    %3 = VPUIP.Copy inputs(%1 : memref<1x70x28x27xf16, #NHWC, @DDR>) outputs(%2 : memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>) -> memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    %4 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 10, 28, 27] : memref<1x70x28x27xf16, #NHWC, @DDR> to memref<1x10x28x27xf16, {order = #NHWC, strides = [52920, 1, 1890, 70]}, @DDR>
    %5 = VPUIP.SubView %buf1 [0, 70, 0, 0] [1, 10, 28, 27] : memref<1x80x28x27xf16, #NHWC, @DDR> to memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    %6 = VPUIP.Copy inputs(%4 : memref<1x10x28x27xf16, {order = #NHWC, strides = [52920, 1, 1890, 70]}, @DDR>) outputs(%5 : memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>) -> memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    %7 = VPUIP.ConcatView inputs(%3, %6 : memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>) outputs(%buf1 : memref<1x80x28x27xf16, #NHWC, @DDR>) -> memref<1x80x28x27xf16, #NHWC, @DDR>
    return %7 : memref<1x80x28x27xf16, #NHWC, @DDR>

    // Copy %3 with parent Copy %1 will be tried to be optimized
    // do not optimize since Copy %1 is also used by SubView %4
    // and will not be removed.
    // implement a solution to optimize both E#35612

    // currently no changes after pass, desired outcome in E#35612

    // CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x70x28x27xf16, #NHWC, @DDR>
    // CHECK:       [[VAR1:%.*]] = memref.alloc() : memref<1x80x28x27xf16, #NHWC, @DDR>
    // CHECK:       [[VAR2:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 1] [1, 70, 28, 27] :
    // CHECK-SAME:      memref<1x80x28x28xf16, #NHWC, @DDR> to memref<1x70x28x27xf16, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>
    // CHECK:       [[VAR3:%.*]] = VPUIP.Copy inputs([[VAR2]] : memref<1x70x28x27xf16, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>) outputs([[VAR0]] : memref<1x70x28x27xf16, #NHWC, @DDR>) ->
    // CHECK-SAME:      memref<1x70x28x27xf16, #NHWC, @DDR>
    // CHECK:       [[VAR4:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 0] [1, 70, 28, 27] :
    // CHECK-SAME:      memref<1x80x28x27xf16, #NHWC, @DDR> to memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    // CHECK:       [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x70x28x27xf16, #NHWC, @DDR>) outputs([[VAR4]] : memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>) ->
    // CHECK-SAME:      memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    // CHECK:       [[VAR6:%.*]] = VPUIP.SubView [[VAR3]] [0, 0, 0, 0] [1, 10, 28, 27] :
    // CHECK-SAME:      memref<1x70x28x27xf16, #NHWC, @DDR> to memref<1x10x28x27xf16, {order = #NHWC, strides = [52920, 1, 1890, 70]}, @DDR>
    // CHECK:       [[VAR7:%.*]] = VPUIP.SubView [[VAR1]] [0, 70, 0, 0] [1, 10, 28, 27] :
    // CHECK-SAME:      memref<1x80x28x27xf16, #NHWC, @DDR> to memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    // CHECK:       [[VAR8:%.*]] = VPUIP.Copy inputs([[VAR6]] : memref<1x10x28x27xf16, {order = #NHWC, strides = [52920, 1, 1890, 70]}, @DDR>) outputs([[VAR7]] : memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>) ->
    // CHECK-SAME:      memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    // CHECK:       [[VAR9:%.*]] = VPUIP.ConcatView inputs([[VAR5]], [[VAR8]] : memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x80x28x27xf16, #NHWC, @DDR>) -> memref<1x80x28x27xf16, #NHWC, @DDR>
    // CHECK:       return %9 : memref<1x80x28x27xf16, #NHWC, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @FuseLastCopiesChainWithNCEClusterTiling(%arg0: memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR> {
    %cst = const.Declare memref<1x8x2x2xf16, @DDR> = dense<1.000000e+00> : tensor<1x16x2x2xf16>, [#const.SubView<[0, 0, 0, 0], [1, 8, 2, 2]>]

    %0 = memref.alloc() : memref<1x8x2x2xf16, @DDR>
    %1 = VPUIP.NCEClusterTiling
            inputs(%cst as %arg1: memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>)
            outputs(%0 as %arg2: memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR> {
      %5 = VPUIP.Copy inputs(%arg1 : memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>) outputs(%arg2 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>
    }

    %2 = memref.alloc() : memref<1x8x2x2xf16, @DDR>
    %3 = VPUIP.Copy inputs(%1 : memref<1x8x2x2xf16, @DDR>) outputs(%2 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>
    %4 = VPUIP.Copy inputs(%3 : memref<1x8x2x2xf16, @DDR>) outputs(%arg0 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

   return %4 : memref<1x8x2x2xf16, @DDR>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<1x8x2x2xf16, @DDR> = dense<1.000000e+00> : tensor<1x16x2x2xf16>, [#const.SubView<[0, 0, 0, 0], [1, 8, 2, 2]>]
    // CHECK:   [[VAR0:%.*]] = VPUIP.NCEClusterTiling inputs([[CST]] as %arg1: memref<1x8x2x2xf16, {order = #NCHW, strides = [64, 4, 2, 1]}, @DDR>)
    // CHECK-SAME:                                       outputs(%arg0 as %arg2: memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR> {
    // CHECK:                   VPUIP.Copy
    // CHECK:   }

    // CHECK:   return [[VAR0]] : memref<1x8x2x2xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @FuseTwoLastCopies(%arg0: memref<1x8x2x2xf16, @DDR>, %arg1: memref<1x32x2x2xf16, @DDR>) -> (memref<1x8x2x2xf16, @DDR>, memref<1x32x2x2xf16, @DDR>) {
    %cst0 = const.Declare memref<1x8x2x2xf16, @DDR> = dense<1.000000e+00> : tensor<1x16x2x2xf16>, [#const.SubView<[0, 0, 0, 0], [1, 8, 2, 2]>]
    %cst1 = const.Declare memref<1x32x2x2xf16, @DDR> = dense<2.000000e+00> : tensor<1x32x2x2xf16>

    %1 = memref.alloc() : memref<1x8x2x2xf16, @DDR>
    %2 = VPUIP.Copy inputs(%cst0 : memref<1x8x2x2xf16, @DDR>) outputs(%1 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>
    %3 = VPUIP.Copy inputs(%2 : memref<1x8x2x2xf16, @DDR>) outputs(%arg0 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

    %4 = memref.alloc() : memref<1x32x2x2xf16, @DDR>
    %5 = VPUIP.Copy inputs(%cst1 : memref<1x32x2x2xf16, @DDR>) outputs(%4 : memref<1x32x2x2xf16, @DDR>) -> memref<1x32x2x2xf16, @DDR>
    %6 = VPUIP.Copy inputs(%5 : memref<1x32x2x2xf16, @DDR>) outputs(%arg1 : memref<1x32x2x2xf16, @DDR>) -> memref<1x32x2x2xf16, @DDR>

    return %3, %6 : memref<1x8x2x2xf16, @DDR>, memref<1x32x2x2xf16, @DDR>

    // CHECK-DAG:   [[CST0:%.*]] = const.Declare memref<1x32x2x2xf16, @DDR> = dense<2.000000e+00> : tensor<1x32x2x2xf16>
    // CHECK-DAG:   [[CST1:%.*]] = const.Declare memref<1x8x2x2xf16, @DDR> = dense<1.000000e+00> : tensor<1x16x2x2xf16>, [#const.SubView<[0, 0, 0, 0], [1, 8, 2, 2]>]

    // CHECK:   [[VAR0:%.*]] = VPUIP.Copy inputs([[CST1]] : memref<1x8x2x2xf16, @DDR>) outputs(%arg0 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>
    // CHECK:   [[VAR1:%.*]] = VPUIP.Copy inputs([[CST0]] : memref<1x32x2x2xf16, @DDR>) outputs(%arg1 : memref<1x32x2x2xf16, @DDR>) -> memref<1x32x2x2xf16, @DDR>

    // CHECK:   return [[VAR0]], [[VAR1]] : memref<1x8x2x2xf16, @DDR>, memref<1x32x2x2xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @CopyWithQuantizeCastOp(%arg0: memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>,
                            %arg1: memref<1x32x56x56xui8, #NHWC>) -> memref<1x32x56x56xui8, #NHWC> {

    %ddr_buf = memref.alloc() : memref<1x32x56x56x!qElemType, #NHWC, @DDR>

    %0 = VPUIP.SubView %ddr_buf [0, 0, 0, 0] [1, 16, 56, 56] :
        memref<1x32x56x56x!qElemType, #NHWC, @DDR> to
        memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
        -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>

    %2 = VPUIP.SubView %ddr_buf [0, 16, 0, 0] [1, 16, 56, 56] :
        memref<1x32x56x56x!qElemType, #NHWC, @DDR> to
        memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
    %3 = VPUIP.Copy
        inputs(%arg0 : memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>)
        outputs(%2 : memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>)
        -> memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>

    %4 = VPUIP.ConcatView
        inputs(%1, %3 :
            memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>,
            memref<1x16x56x56x!qElemType, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @DDR>
        )
        outputs(%ddr_buf : memref<1x32x56x56x!qElemType, #NHWC, @DDR>)
        -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>
    %5 = VPUIP.QuantizeCast
        inputs(%4 : memref<1x32x56x56x!qElemType, #NHWC, @DDR>)
        -> memref<1x32x56x56xui8, #NHWC, @DDR>
    %6 = VPUIP.Copy
        inputs(%5 : memref<1x32x56x56xui8, #NHWC, @DDR>)
        outputs(%arg1 : memref<1x32x56x56xui8, #NHWC>)
        -> memref<1x32x56x56xui8, #NHWC>
    return %6 : memref<1x32x56x56xui8, #NHWC>

    // verify that the SubView operation is not removed along with the copy operation

    // CHECK:       [[VAL0:%.+]] = VPUIP.QuantizeCast  inputs(%arg1 : memref<1x32x56x56xui8, #NHWC>) -> memref<1x32x56x56x!qElemType, #NHWC, @DDR>

    // CHECK:       [[VAL1:%.+]] = VPUIP.SubView [[VAL0]]
    // CHECK:       VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0
    // CHECK-SAME:      outputs([[VAL1]]
    // CHECK:       [[VAL2:%.+]] = VPUIP.SubView [[VAL0]]
    // CHECK:       VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0
    // CHECK-SAME:      outputs([[VAL2]]

    // copy optimized
    // CHECK-NOT:   VPUIP.ConcatView
    // CHECK-NOT:   VPUIP.QuantizeCast
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:       return %arg1 : memref<1x32x56x56xui8, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @CopyWithSubViewOp(%in : memref<1x16x113x113xf16, #NHWC, @DDR>,
                        %weight_table : memref<16x1x1x4xsi32, @CMX_NN>,
                        %act_wind : memref<16x1x1x16xui8, @CMX_NN>)
                        -> memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN> {

    %buf0 = memref.alloc() : memref<1x16x113x113xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x32x56x56xf16, #NHWC, @CMX_NN>

    // activation copy-in
    %0 = VPUIP.Copy
            inputs(%in : memref<1x16x113x113xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x16x113x113xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x113x113xf16, #NHWC, @CMX_NN>

    %1 = VPUIP.NCEClusterTask
        {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [3, 3],
            kernel_strides = [2, 2],
            task_type = "MAXPOOL"
        }
        input(%0 : memref<1x16x113x113xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_wind : memref<16x1x1x16xui8, @CMX_NN>)
        parent_input(%0 : memref<1x16x113x113xf16, #NHWC, @CMX_NN>)
        parent_output(%buf1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        outputs(%buf1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 10, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    // slice of buffer where the NCE writes
    %2 = VPUIP.SubView %buf2 [0, 0, 0, 0] [1, 16, 56, 56] :
        memref<1x32x56x56xf16, #NHWC, @CMX_NN> to
        memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    // copy of the output NCE from NNCMX->NNCMX
    %3 = VPUIP.Copy
        inputs(%1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        outputs(%2 : memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>)
        -> memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    return %2 : memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    // verify that the SubView operation is not removed along with the copy operation

    // CHECK:       [[VAL0:%.+]] = memref.alloc() : memref<1x16x113x113xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAL1:%.+]] = memref.alloc() : memref<1x32x56x56xf16, #NHWC, @CMX_NN>

    // CHECK:       [[VAL2:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x16x113x113xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[VAL0]] : memref<1x16x113x113xf16, #NHWC, @CMX_NN>)

    // subView present
    // CHECK:       [[VAL3:%.+]] = VPUIP.SubView [[VAL1]] [0, 0, 0, 0] [1, 16, 56, 56] : memref<1x32x56x56xf16, #NHWC, @CMX_NN> to
    // CHECK-SAME:      memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

    // CHECK:       [[VAL4:%.+]] = VPUIP.NCEClusterTask
    // CHECK:           input([[VAL2]] : memref<1x16x113x113xf16, #NHWC, @CMX_NN>)
    // CHECK:           outputs([[VAL3]] : memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>)

    // copy optimized
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:       return [[VAL3:%.+]] : memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDR2DDRCopyOutput(%in : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                        -> memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> {
    // Input Tile 1
    %0 = memref.alloc() : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %1 = memref.alloc() : memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>

    %2 = VPUIP.NCEClusterTiling inputs(%in as %arg2: memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
            outputs(%0 as %arg3: memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>)
                -> memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> {
        %729 = VPUIP.Copy inputs(%arg2 : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                outputs(%arg3 : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>)
                        -> memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>
      }
    %3 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 32, 64, 128] : memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
            to memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>
    %4 = VPUIP.Copy inputs(%2 : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
            outputs(%3 : memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>)
                -> memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>

    // Input Tile 2
    %5 = memref.alloc() : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %6 = VPUIP.NCEClusterTiling inputs(%in as %arg2: memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
            outputs(%5 as %arg3: memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>)
                -> memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> {
        %729 = VPUIP.Copy inputs(%arg2 : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
            outputs(%arg3 : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>)
                    -> memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>
      }
    %7 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 32, 64, 128] : memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
            to memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>
    %8 = VPUIP.Copy inputs(%6 : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
            outputs(%7 : memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>)
                    -> memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>

    // Concat
    %9 = VPUIP.ConcatView inputs(%4, %8 :
            memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>,
            memref<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [524288, 1, 4096, 32]}, @DDR>)
            outputs(%1 : memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
                -> memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    return %9 : memref<1x32x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>

    // CHECK:       [[BUFF:%.*]] = memref.alloc() : memref<1x32x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView [[BUFF]] [0, 0, 0, 0] [1, 32, 64, 128] :
    // CHECK-SAME:      memref<1x32x128x128xf16, #NHWC, @DDR> to memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>
    // CHECK:       [[COPY_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg1: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_1]] as %arg2: memref<1x32x64x128xf16, #NHWC>)
    // CHECK-SAME:          -> memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR> {
    // CHECK:       [[COPY_1_INNER:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs(%arg2 : memref<1x32x64x128xf16, #NHWC>)
    // CHECK-SAME:          -> memref<1x32x64x128xf16, #NHWC>

    // CHECK:       [[SUBVIEW_2:%.*]] = VPUIP.SubView [[BUFF]] [0, 0, 64, 0] [1, 32, 64, 128] :
    // CHECK-SAME:      memref<1x32x128x128xf16, #NHWC, @DDR> to memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>
    // CHECK:       [[COPY_2:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg1: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_2]] as %arg2: memref<1x32x64x128xf16, #NHWC>)
    // CHECK-SAME:          -> memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR> {
    // CHECK:       [[COPY_2_INNER:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs(%arg2 : memref<1x32x64x128xf16, #NHWC>)
    // CHECK-SAME:          -> memref<1x32x64x128xf16, #NHWC>

    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_1]], [[COPY_2]] : memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>, memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>)
    // CHECK-SAME:      outputs([[BUFF]] : memref<1x32x128x128xf16, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x32x128x128xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCAT]] : memref<1x32x128x128xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

func.func @DDR2DDRCopyInput(%in : memref<1x144x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>,
                       %weights: memref<32x144x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>,
                       %weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
                        -> !OutputDistributed {
    %0 = VPUIP.SubView %in [0, 0, 0, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %1 = memref.alloc() : memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %2 = VPUIP.Copy
            inputs(%0 : memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>)
            outputs(%1 : memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
                -> memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %4 = VPUIP.NCEClusterTiling
            inputs(%2 as %arg2: memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>)
            outputs(%3 as %arg3: memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                -> !OutputDistributed {
        %inner = VPUIP.Copy
                inputs(%arg2 : memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>)
                outputs(%arg3 : memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                    -> memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>
    }
    %5 = VPURT.AllocDistributed -> !OutputDistributed
    %6 = VPUIP.NCEClusterTiling
            inputs(
                %4 as %arg2: memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>,
                %weights as %arg3: memref<32x144x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>,
                %weights_table as %arg4: memref<32x1x1x4xsi32, @CMX_NN>)
            outputs(
                %5 as %arg5: memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                    -> !OutputDistributed {
        %inner = VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = "CONV"}
            input(
                %arg2 : memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                weights(%arg3 : memref<32x144x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                weight_table(%arg4 : memref<32x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                parent_output(%arg5 : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                outputs(%arg5 : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                    -> memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
        } PPE :  {
        }
    }
    return %6 : !OutputDistributed

    // CHECK:       [[SUBVIEW:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[BUFFER_1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW]] as %arg3: memref<1x144x64x128xf16, #NHWC>)
    // CHECK-SAME:      outputs(%1 as %arg4: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    // CHECK:       [[COPY_INNER:%.*]] = VPUIP.Copy inputs(%arg3 : memref<1x144x64x128xf16, #NHWC>)
    // CHECK-SAME:      outputs(%arg4 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFFER_2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[NCE:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[COPY]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg1 as %arg4: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg2 as %arg5: memref<32x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs(%3 as %arg6: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       return [[NCE]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = !quant.uniform<u8:f16, 5.7832517137714463:123>
!qElemType1 = !quant.uniform<u8:f16, 7.1335556927849266:124>

func.func @ParallelDDR2DDRCopyOutput(%in0 : memref<1x64x48x88x!qElemType0, #NHWC, @CMX_NN>,
                                %in1 : memref<1x64x48x88x!qElemType0, #NHWC, @CMX_NN>)
                                    -> memref<1x64x48x176x!qElemType1, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTiling inputs(%in0 as %arg2: memref<1x64x48x88x!qElemType0, #NHWC, @CMX_NN>,
                                       %in1 as %arg3: memref<1x64x48x88x!qElemType0, #NHWC, @CMX_NN>)
                                outputs(%0 as %arg4: memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
                                    -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
        %1232 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, minimumHardwareExecutionCost = 1081 : i64, task_type = "ELTWISE"}
            input(%arg2 : memref<1x64x48x88x!qElemType0, #NHWC, @CMX_NN>)
            weights(%arg3 : memref<1x64x48x88x!qElemType0, #NHWC, @CMX_NN>)
            parent_input(%arg2 : memref<1x64x48x88x!qElemType0, #NHWC, @CMX_NN>)
            parent_output(%arg4 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
            outputs(%arg4 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask "ADD" {clamp_high = 131 : i64, clamp_low = -124 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [26565], quant_post_shift = 0 : i64, quant_shift = [15]}
        }
    }

    %2 = memref.alloc() : memref<1x64x48x88x!qElemType1, #NHWC, @DDR>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
                                outputs(%2 as %arg3: memref<1x64x48x88x!qElemType1, #NHWC>)
                                    -> memref<1x64x48x88x!qElemType1, #NHWC, @DDR> {
        %1232 = VPUIP.Copy inputs(%arg2 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
                           outputs(%arg3 : memref<1x64x48x88x!qElemType1, #NHWC>)
                               -> memref<1x64x48x88x!qElemType1, #NHWC>
    }

    %4 = memref.alloc() : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>
    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 64, 48, 88] [1, 1, 1, 2] : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>
            to memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    %6 = VPUIP.Copy inputs(%3 : memref<1x64x48x88x!qElemType1, #NHWC, @DDR>)
                    outputs(%5 : memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)
                        -> memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    %7 = VPUIP.SubView %4 [0, 0, 0, 1] [1, 64, 48, 88] [1, 1, 1, 2] : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>
            to memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    %8 = VPUIP.Copy inputs(%3 : memref<1x64x48x88x!qElemType1, #NHWC, @DDR>)
                    outputs(%7 : memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)
                        -> memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>

    %9 = VPUIP.ConcatView inputs(%6, %8 :
                                 memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>,
                                 memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)
                          outputs(%4 : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>)
                              -> memref<1x64x48x176x!qElemType1, #NHWC, @DDR>
    return %9 : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>
    // CHECK:       [[ADD_0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg2: memref<1x64x48x88x!qElemType0, #NHWC, @CMX_NN>,
    // CHECK-SAME:             %arg1 as %arg3: memref<1x64x48x88x!qElemType0, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFF_0]] as %arg4: memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    // CHECK:           [[ADD_0_INNER:%.*]] = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, minimumHardwareExecutionCost = 1081 : i64, task_type = "ELTWISE"}
    // CHECK-SAME:          input(%arg2 : memref<1x64x48x88x!qElemType0, #NHWC, @CMX_NN>)
    // CHECK-SAME:          weights(%arg3 : memref<1x64x48x88x!qElemType0, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_input(%arg2 : memref<1x64x48x88x!qElemType0, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output(%arg4 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg4 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:              -> memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN> variants :  {
    // CHECK:                   DPUTask {cluster_id = 0 : i64, mpe_mode = "MATRIX", outEnd = [87, 47, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    // CHECK:           } PPE :  {
    // CHECK:                   PPETask "ADD" {clamp_high = 131 : i64, clamp_low = -124 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [26565], quant_post_shift = 0 : i64, quant_shift = [15]}
    // CHECK:           }
    // CHECK:       }

    // CHECK:       [[BUFF_1:%.*]] = memref.alloc() : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 64, 48, 88] [1, 1, 1, 2] :
    // CHECK-SAME:      memref<1x64x48x176x!qElemType1, #NHWC, @DDR> to memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    // CHECK:       [[COPY_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[ADD_0]] as %arg2: memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_1]] as %arg3: memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)
    // CHECK-SAME:          -> memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR> {
    // CHECK:           [[COPY_1_INNER:%.*]] = VPUIP.Copy
    // CHECK-SAME:          inputs(%arg2 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg3 : memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)
    // CHECK-SAME:          -> memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    // CHECK:       }
    // CHECK:       [[SUBVIEW_2:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 1] [1, 64, 48, 88] [1, 1, 1, 2] :
    // CHECK-SAME:      memref<1x64x48x176x!qElemType1, #NHWC, @DDR> to memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    // CHECK:       [[COPY_2:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[ADD_0]] as %arg2: memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_2]] as %arg3: memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)
    // CHECK-SAME:          -> memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR> {
    // CHECK:           [[COPY_2_INNER:%.*]] = VPUIP.Copy
    // CHECK-SAME:          inputs(%arg2 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg3 : memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)
    // CHECK-SAME:              -> memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>
    // CHECK:       }

    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_1]], [[COPY_2]] : memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>,
    // CHECK-SAME:                                      memref<1x64x48x88x!qElemType1, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_1]] : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x64x48x176x!qElemType1, #NHWC, @DDR>

    // CHECK:       return [[CONCAT]] : memref<1x64x48x176x!qElemType1, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ParallelDDR2DDRCopyOutputNoChangeToFixAccuracy(%in0 : !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
                                        %in1 : !VPUIP.DistributedBuffer<80x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                                        %in2 : !VPUIP.DistributedBuffer<80x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                                        %in3 : !VPUIP.DistributedBuffer<1x1x1x16xui8, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                                            -> memref<1x80x66x64xf16, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTiling inputs(%in0 as %arg2: memref<1x80x64x64xf16, #NHWC, @CMX_NN>, %in1 as %arg3: memref<80x16x1x1xf16, #NHWC, @CMX_NN>, %in2 as %arg4: memref<80x1x1x4xsi32, @CMX_NN>, %in3 as %arg5: memref<1x1x1x16xui8, @CMX_NN>) outputs(%0 as %arg6: memref<1x80x64x64xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %48 = VPUIP.NCEClusterTask {activation_window_channel_length = 4 : i64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 16540 : i64, task_type = "DWCONV"} input(%arg2 : memref<1x80x64x64xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<80x16x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<80x1x1x4xsi32, @CMX_NN>) activation_window(%arg5 : memref<1x1x1x16xui8, @CMX_NN>) parent_input(%arg2 : memref<1x80x64x64xf16, #NHWC, @CMX_NN>) parent_output(%arg6 : memref<1x80x64x64xf16, #NHWC, @CMX_NN>) outputs(%arg6 : memref<1x80x64x64xf16, #NHWC, @CMX_NN>) -> memref<1x80x64x64xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [63, 31, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [63, 31, 79], outStart = [0, 0, 64], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [63, 63, 63], outStart = [0, 32, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
        DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [63, 63, 79], outStart = [0, 32, 64], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
      } PPE : {
        PPETask "LRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
      }
    }

    %2 = memref.alloc() : memref<1x80x64x64xf16, #NHWC, @DDR>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x80x64x64xf16, #NHWC, @CMX_NN>) outputs(%2 as %arg3: memref<1x80x64x64xf16, #NHWC>) -> memref<1x80x64x64xf16, #NHWC, @DDR> {
      %48 = VPUIP.Copy inputs(%arg2 : memref<1x80x64x64xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x80x64x64xf16, #NHWC>) -> memref<1x80x64x64xf16, #NHWC>
    }
    %4 = VPUIP.SubView %3 [0, 0, 1, 0] [1, 80, 1, 64] : memref<1x80x64x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>
    %5 = VPUIP.SubView %3 [0, 0, 62, 0] [1, 80, 1, 64] : memref<1x80x64x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>
    %6 = memref.alloc() : memref<1x80x66x64xf16, #NHWC, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 80, 1, 64] : memref<1x80x66x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    %8 = VPUIP.Copy inputs(%4 : memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>) outputs(%7 : memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>) -> memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    %9 = VPUIP.SubView %6 [0, 0, 1, 0] [1, 80, 64, 64] : memref<1x80x66x64xf16, #NHWC, @DDR> to memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    %10 = VPUIP.Copy inputs(%3 : memref<1x80x64x64xf16, #NHWC, @DDR>) outputs(%9 : memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>) -> memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    %11 = VPUIP.SubView %6 [0, 0, 65, 0] [1, 80, 1, 64] : memref<1x80x66x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    %12 = VPUIP.Copy inputs(%5 : memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>) outputs(%11 : memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>) -> memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    %13 = VPUIP.ConcatView inputs(%8, %10, %12 : memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>, memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>, memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>) outputs(%6 : memref<1x80x66x64xf16, #NHWC, @DDR>) -> memref<1x80x66x64xf16, #NHWC, @DDR>

    return %13 : memref<1x80x66x64xf16, #NHWC, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[NCETASK_0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg4: memref<1x80x64x64xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:             %arg1 as %arg5: memref<80x16x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:             %arg2 as %arg6: memref<80x1x1x4xsi32, @CMX_NN>,
    // CHECK-SAME:             %arg3 as %arg7: memref<1x1x1x16xui8, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFF_0]] as %arg8: memref<1x80x64x64xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x80x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           [[INNER_0:%.*]]  = VPUIP.NCEClusterTask {activation_window_channel_length = 4 : i64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 16540 : i64, task_type = "DWCONV"}
    // CHECK-SAME:          input(%arg4 : memref<1x80x64x64xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME           weights(%arg5 : memref<80x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          weight_table(%arg6 : memref<80x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:          activation_window(%arg7 : memref<1x1x1x16xui8, @CMX_NN>)
    // CHECK-SAME:          parent_input(%arg4 : memref<1x80x64x64xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output(%arg8 : memref<1x80x64x64xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg8 : memref<1x80x64x64xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:              -> memref<1x80x64x64xf16, #NHWC, @CMX_NN> variants : {
    // CHECK:                   DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [63, 31, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    // CHECK:                   DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [63, 31, 79], outStart = [0, 0, 64], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    // CHECK:                   DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [63, 63, 63], outStart = [0, 32, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    // CHECK:                   DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [63, 63, 79], outStart = [0, 32, 64], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    // CHECK:           } PPE : {
    // CHECK:                   PPETask "LRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    // CHECK:           }
    // CHECK:       }

    // CHECK:       [[BUFF_1:%.*]] = memref.alloc() : memref<1x80x64x64xf16, #NHWC, @DDR>
    // CHECK:       [[COPY_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[NCETASK_0]] as %arg4: memref<1x80x64x64xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFF_1]] as %arg5: memref<1x80x64x64xf16, #NHWC>)
    // CHECK-SAME:          -> memref<1x80x64x64xf16, #NHWC, @DDR> {
    // CHECK:           [[COPY_1_INNER:%.*]] = VPUIP.Copy
    // CHECK-SAME:          inputs(%arg4 : memref<1x80x64x64xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg5 : memref<1x80x64x64xf16, #NHWC>)
    // CHECK-SAME:          -> memref<1x80x64x64xf16, #NHWC>
    // CHECK:       }
    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView [[COPY_1]] [0, 0, 1, 0] [1, 80, 1, 64] :
    // CHECK-SAME:      memref<1x80x64x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>
    // CHECK:       [[SUBVIEW_2:%.*]] = VPUIP.SubView [[COPY_1]] [0, 0, 62, 0] [1, 80, 1, 64] :
    // CHECK-SAME:      memref<1x80x64x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>

    // CHECK:       [[BUFF_2:%.*]] = memref.alloc() : memref<1x80x66x64xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_3:%.*]] = VPUIP.SubView [[BUFF_2]] [0, 0, 0, 0] [1, 80, 1, 64] :
    // CHECK-SAME:      memref<1x80x66x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    // CHECK:       [[COPY_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_1]] : memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_3]] : memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>)
    // CHECK-SAME:          -> memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>

    // CHECK:       [[SUBVIEW_4:%.*]] = VPUIP.SubView [[BUFF_2]] [0, 0, 1, 0] [1, 80, 64, 64] :
    // CHECK-SAME:      memref<1x80x66x64xf16, #NHWC, @DDR> to memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    // CHECK:       [[COPY_3:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[COPY_1]] : memref<1x80x64x64xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_4]] : memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>)
    // CHECK-SAME:          -> memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>

    // CHECK:       [[SUBVIEW_5:%.*]] = VPUIP.SubView [[BUFF_2]] [0, 0, 65, 0] [1, 80, 1, 64] :
    // CHECK-SAME:      memref<1x80x66x64xf16, #NHWC, @DDR> to memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>
    // CHECK:       [[COPY_4:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_2]] : memref<1x80x1x64xf16, {order = #NHWC, strides = [327680, 1, 5120, 80]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_5]] : memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>)
    // CHECK-SAME:          -> memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>

    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_2]], [[COPY_3]], [[COPY_4]] : memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>,
    // CHECK-SAME:                                                  memref<1x80x64x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>,
    // CHECK-SAME:                                                  memref<1x80x1x64xf16, {order = #NHWC, strides = [337920, 1, 5120, 80]}, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_2]] : memref<1x80x66x64xf16, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x80x66x64xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCAT]] : memref<1x80x66x64xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>

func.func @DDR2DDRCopyOutputNOSubview(%in0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
                                %in1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                    -> memref<1x64x48x88x!qElemType, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTiling inputs(%in0 as %arg2: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
                                       %in1 as %arg3: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                outputs(%0 as %arg4: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                    -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
        %1232 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, minimumHardwareExecutionCost = 1081 : i64, task_type = "ELTWISE"}
            input(%arg2 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            weights(%arg3 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_input(%arg2 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_output(%arg4 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            outputs(%arg4 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask "ADD" {clamp_high = 131 : i64, clamp_low = -124 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [26565], quant_post_shift = 0 : i64, quant_shift = [15]}
        }
    }
    %2 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                outputs(%2 as %arg3: memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
                                    -> memref<1x64x48x88x!qElemType, #NHWC, @DDR> {
        %1232 = VPUIP.Copy inputs(%arg2 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                           outputs(%arg3 : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
                               -> memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    }
    %4 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    %5 = VPUIP.Copy inputs(%3 : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
            outputs(%4 : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
            -> memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    %6 = VPUIP.ConcatView inputs(%5 : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
            outputs(%4 : memref<1x64x48x88x!qElemType,#NHWC, @DDR>) -> memref<1x64x48x88x!qElemType, #NHWC, @DDR>

    return %6 : memref<1x64x48x88x!qElemType, #NHWC, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[ADD_0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg2: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
    // CHECK-SAME:             %arg1 as %arg3: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFF_0]] as %arg4: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    // CHECK:           [[ADD_0_INNER:%.*]] = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, minimumHardwareExecutionCost = 1081 : i64, task_type = "ELTWISE"}
    // CHECK-SAME:          input(%arg2 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          weights(%arg3 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_input(%arg2 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output(%arg4 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg4 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:              -> memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN> variants :  {
    // CHECK:                   DPUTask {cluster_id = 0 : i64, mpe_mode = "MATRIX", outEnd = [87, 47, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    // CHECK:           } PPE :  {
    // CHECK:                   PPETask "ADD" {clamp_high = 131 : i64, clamp_low = -124 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [26565], quant_post_shift = 0 : i64, quant_shift = [15]}
    // CHECK:           }
    // CHECK:       }
    // CHECK:       [[BUFF_1:%.*]] = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    // CHECK:       [[COPY_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[ADD_0]] as %arg2: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFF_1]] as %arg3: memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x64x48x88x!qElemType, #NHWC, @DDR> {
    // CHECK:           [[COPY_1_INNER:%.*]] = VPUIP.Copy
    // CHECK-SAME:          inputs(%arg2 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs(%arg3 : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    // CHECK:       }
    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_1]] : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_1]] : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x64x48x88x!qElemType, #NHWC, @DDR>

    // CHECK:       return [[CONCAT]] : memref<1x64x48x88x!qElemType, #NHWC, @DDR>
}

// -----

//  Optimize the left DDR2DDR copy in below case:
//                      ConcatView                                                     ConcatView
//                     /          \                                                  /          \
//          Copy(DDR2DDR)      SubView                                              |        SubView
//                  \             |                                                 |            |
//                   \        Copy(DDR2DDR)      =>                                 |       Copy(DDR2DDR)
//                    \        /                                                     \        /
//                        |                                                               |
//                        |                                                               |
//                    ConcatView                                                       ConcatView
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.99909667968750004:124>

func.func @DDR2DDROfConcatInput(%arg0: memref<1x512x18x32x!qElemType, #NHWC, @DDR>) -> memref<1x512x19x33x!qElemType, #NHWC, @DDR> {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 31] [1, 512, 18, 1] : memref<1x512x18x32x!qElemType, #NHWC, @DDR> to memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>

    %1 = memref.alloc() : memref<1x512x18x33x!qElemType, #NHWC, @DDR>

    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 512, 18, 32] : memref<1x512x18x33x!qElemType, #NHWC, @DDR> to memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>
    %3 = VPUIP.Copy inputs(%arg0 : memref<1x512x18x32x!qElemType, #NHWC, @DDR>) outputs(%2 : memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %4 = VPUIP.SubView %1 [0, 0, 0, 32] [1, 512, 18, 1] : memref<1x512x18x33x!qElemType, #NHWC, @DDR> to memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>
    %5 = VPUIP.Copy inputs(%0 : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>) outputs(%4 : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %6 = VPUIP.ConcatView inputs(%3, %5 : memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) outputs(%1 : memref<1x512x18x33x!qElemType, #NHWC, @DDR>) -> memref<1x512x18x33x!qElemType, #NHWC, @DDR>

    %7 = VPUIP.SubView %6 [0, 0, 17, 0] [1, 512, 1, 33] : memref<1x512x18x33x!qElemType, #NHWC, @DDR> to memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %8 = memref.alloc() : memref<1x512x19x33x!qElemType, #NHWC, @DDR>

    %9 = VPUIP.SubView %8 [0, 0, 0, 0] [1, 512, 18, 33] : memref<1x512x19x33x!qElemType, #NHWC, @DDR> to memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>
    %10 = VPUIP.Copy inputs(%6 : memref<1x512x18x33x!qElemType, #NHWC, @DDR>) outputs(%9 : memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    %11 = VPUIP.SubView %8 [0, 0, 18, 0] [1, 512, 1, 33] : memref<1x512x19x33x!qElemType, #NHWC, @DDR> to memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>
    %12 = VPUIP.Copy inputs(%7 : memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) outputs(%11 : memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>) -> memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    %13 = VPUIP.ConcatView inputs(%10, %12 : memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>) outputs(%8 : memref<1x512x19x33x!qElemType, #NHWC, @DDR>) -> memref<1x512x19x33x!qElemType, #NHWC, @DDR>
    return %13 : memref<1x512x19x33x!qElemType, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_FOR_COPY_0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 31] [1, 512, 18, 1] :
    // CHECK-SAME:      memref<1x512x18x32x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x512x19x33x!qElemType, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 512, 18, 33] :
    // CHECK-SAME:      memref<1x512x19x33x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_0_1:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 0] [1, 512, 18, 32] :
    // CHECK-SAME:      memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR> to
    // CHECK-SAME:      memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_FROM_CONCAT_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x512x18x32x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_1]] : memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_0_2:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 32] [1, 512, 18, 1] :
    // CHECK-SAME:      memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR> to
    // CHECK-SAME:      memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_FROM_SUBVIEW:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_FOR_COPY_0]] : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_2]] : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[CONCAT_PARENT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_FROM_CONCAT_0]], [[COPY_FROM_SUBVIEW]] :
    // CHECK-SAME:          memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>,
    // CHECK-SAME:          memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0]] : memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_CONCAT_PARENT_0:%.*]] = VPUIP.SubView [[CONCAT_PARENT]] [0, 0, 17, 0] [1, 512, 1, 33] :
    // CHECK-SAME:      memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR> to
    // CHECK-SAME:      memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_BUFF_0:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 18, 0] [1, 512, 1, 33] :
    // CHECK-SAME:      memref<1x512x19x33x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:       memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_RESULT:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_CONCAT_PARENT_0]] : memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_BUFF_0]] : memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[CONCAT_CHILD:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[CONCAT_PARENT]], [[COPY_RESULT]] :
    // CHECK-SAME:          memref<1x512x18x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>,
    // CHECK-SAME:          memref<1x512x1x33x!qElemType, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_0]] : memref<1x512x19x33x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x512x19x33x!qElemType, #NHWC, @DDR>

    // CHECK:       return [[CONCAT_CHILD]] : memref<1x512x19x33x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDR2DDROfConcatWithConstInput(%arg0: memref<1x512x18x32xi8, #NHWC, @DDR>) -> memref<1x512x19x33xi8, #NHWC, @DDR> {
    %cst = const.Declare memref<1x512x1x33xi8, #NHWC> = dense<0> : tensor<1x512x1x33xi8, {order = #NHWC}>

    %0 = VPUIP.SubView %arg0 [0, 0, 0, 31] [1, 512, 18, 1] : memref<1x512x18x32xi8, #NHWC, @DDR> to memref<1x512x18x1xi8, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>

    %1 = memref.alloc() : memref<1x512x18x33xi8, #NHWC, @DDR>

    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 512, 18, 32] : memref<1x512x18x33xi8, #NHWC, @DDR> to memref<1x512x18x32xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>
    %3 = VPUIP.Copy inputs(%arg0 : memref<1x512x18x32xi8, #NHWC, @DDR>) outputs(%2 : memref<1x512x18x32xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x32xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %4 = VPUIP.SubView %1 [0, 0, 0, 32] [1, 512, 18, 1] : memref<1x512x18x33xi8, #NHWC, @DDR> to memref<1x512x18x1xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>
    %5 = VPUIP.Copy inputs(%0 : memref<1x512x18x1xi8, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>) outputs(%4 : memref<1x512x18x1xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x1xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %6 = VPUIP.ConcatView inputs(%3, %5 : memref<1x512x18x32xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, memref<1x512x18x1xi8, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) outputs(%1 : memref<1x512x18x33xi8, #NHWC, @DDR>) -> memref<1x512x18x33xi8, #NHWC, @DDR>

    %7 = memref.alloc() : memref<1x512x19x33xi8, #NHWC, @DDR>

    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 512, 18, 33] : memref<1x512x19x33xi8, #NHWC, @DDR> to memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>
    %9 = VPUIP.Copy inputs(%6 : memref<1x512x18x33xi8, #NHWC, @DDR>) outputs(%8 : memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    %10 = VPUIP.SubView %7 [0, 0, 18, 0] [1, 512, 1, 33] : memref<1x512x19x33xi8, #NHWC, @DDR> to memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>
    %11 = VPUIP.Copy inputs(%cst : memref<1x512x1x33xi8, #NHWC>) outputs(%10 : memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>) -> memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    %12 = VPUIP.ConcatView inputs(%9, %11 : memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>, memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>) outputs(%7 : memref<1x512x19x33xi8, #NHWC, @DDR>) -> memref<1x512x19x33xi8, #NHWC, @DDR>
    return %12 : memref<1x512x19x33xi8, #NHWC, @DDR>

    // CHECK-DAG:       [[CONST:%.*]] = const.Declare memref<1x512x1x33xi8, #NHWC> = dense<0> : tensor<1x512x1x33xi8, {order = #NHWC}>

    // CHECK:       [[SUBVIEW_FOR_COPY_0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 31] [1, 512, 18, 1] :
    // CHECK-SAME:      memref<1x512x18x32xi8, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x1xi8, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x512x19x33xi8, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 512, 18, 33] :
    // CHECK-SAME:      memref<1x512x19x33xi8, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_0_1:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 0] [1, 512, 18, 32] :
    // CHECK-SAME:      memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR> to
    // CHECK-SAME:      memref<1x512x18x32xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_FROM_CONCAT_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x512x18x32xi8, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_1]] : memref<1x512x18x32xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x32xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_0_2:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 32] [1, 512, 18, 1] :
    // CHECK-SAME:      memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR> to
    // CHECK-SAME:      memref<1x512x18x1xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_FROM_SUBVIEW:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_FOR_COPY_0]] : memref<1x512x18x1xi8, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_2]] : memref<1x512x18x1xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x1xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[CONCAT_PARENT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_FROM_CONCAT_0]], [[COPY_FROM_SUBVIEW]] :
    // CHECK-SAME:          memref<1x512x18x32xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>,
    // CHECK-SAME:          memref<1x512x18x1xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0]] : memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_BUFF_0:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 18, 0] [1, 512, 1, 33] :
    // CHECK-SAME:      memref<1x512x19x33xi8, #NHWC, @DDR> to
    // CHECK-SAME:       memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_RESULT:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONST]] : memref<1x512x1x33xi8, #NHWC>)
    // CHECK-SAME:      outputs([[SUBVIEW_BUFF_0]] : memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>

    // CHECK:       [[CONCAT_CHILD:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[CONCAT_PARENT]], [[COPY_RESULT]] :
    // CHECK-SAME:          memref<1x512x18x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>,
    // CHECK-SAME:          memref<1x512x1x33xi8, {order = #NHWC, strides = [321024, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_0]] : memref<1x512x19x33xi8, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x512x19x33xi8, #NHWC, @DDR>

    // CHECK:       return [[CONCAT_CHILD]] : memref<1x512x19x33xi8, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDR2DDROfConcatWithConstAndTilingCopyInput(%arg0: memref<1x1x128x200xf16, #NHWC, @DDR>) -> memref<1x1x130x202xf16, #NHWC, @DDR> {
    %cst0 = const.Declare memref<1x1x128x1xf16, #NHWC, @DDR> = dense<0.000000e+00> : tensor<1x1x128x1xf16, {order = #NHWC}>
    %cst1 = const.Declare memref<1x1x128x1xf16, #NHWC, @DDR> = dense<0.000000e+00> : tensor<1x1x128x1xf16, {order = #NHWC}>

    %0 = memref.alloc() : memref<1x1x128x202xf16, #NHWC, @DDR>

    %1 = VPUIP.SubView %0 [0, 0, 0, 1] [1, 1, 128, 200] : memref<1x1x128x202xf16, #NHWC, @DDR>
            to memref<1x1x128x200xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>
    %2 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x1x128x200xf16, #NHWC, @CMX_NN>)
            outputs(%1 as %arg3: memref<1x1x128x200xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>)
                -> memref<1x1x128x200xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR> {
        %1000 = VPUIP.Copy inputs(%arg2 : memref<1x1x128x200xf16, #NHWC, @CMX_NN>)
                outputs(%arg3 : memref<1x1x128x200xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>)
                        -> memref<1x1x128x200xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>
      }

    %3 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 1, 128, 1] : memref<1x1x128x202xf16, #NHWC, @DDR>
            to memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>
    %4 = VPUIP.Copy inputs(%cst0 : memref<1x1x128x1xf16, #NHWC, @DDR>)
            outputs(%3 : memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>)
                -> memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>

    %5 = VPUIP.SubView %0 [0, 0, 0, 201] [1, 1, 128, 1] : memref<1x1x128x202xf16, #NHWC, @DDR>
            to memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>
    %6 = VPUIP.Copy inputs(%cst1 : memref<1x1x128x1xf16, #NHWC, @DDR>)
            outputs(%5 : memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>)
                -> memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>

    %7 = VPUIP.ConcatView inputs(%2, %4, %6 : memref<1x1x128x200xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>,
            memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>,
            memref<1x1x128x1xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>)
            outputs(%0 : memref<1x1x128x202xf16, #NHWC, @DDR>) -> memref<1x1x128x202xf16, #NHWC, @DDR>

    %cst2 = const.Declare memref<1x1x1x202xf16, #NHWC, @DDR> = dense<0.000000e+00> : tensor<1x1x1x202xf16, {order = #NHWC}>
    %cst3 = const.Declare memref<1x1x1x202xf16, #NHWC, @DDR> = dense<0.000000e+00> : tensor<1x1x1x202xf16, {order = #NHWC}>

    %8 = memref.alloc() : memref<1x1x130x202xf16, #NHWC, @DDR>

    %9 = VPUIP.SubView %8 [0, 0, 1, 0] [1, 1, 128, 202] : memref<1x1x130x202xf16, #NHWC, @DDR>
            to memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>
    %10 = VPUIP.Copy inputs(%7 : memref<1x1x128x202xf16, #NHWC, @DDR>)
            outputs(%9 : memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
                -> memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    %11 = VPUIP.SubView %8 [0, 0, 0, 0] [1, 1, 1, 202] : memref<1x1x130x202xf16, #NHWC, @DDR>
            to memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>
    %12 = VPUIP.Copy inputs(%cst2  : memref<1x1x1x202xf16, #NHWC, @DDR>)
            outputs(%11 : memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
                -> memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    %13 = VPUIP.SubView %8 [0, 0, 129, 0] [1, 1, 1, 202] : memref<1x1x130x202xf16, #NHWC, @DDR>
            to memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>
    %14 = VPUIP.Copy inputs(%cst3  : memref<1x1x1x202xf16, #NHWC, @DDR>)
            outputs(%13 : memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
                -> memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    %15 = VPUIP.ConcatView inputs(%10, %12, %14 : memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>,
            memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>,
            memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
            outputs(%8 : memref<1x1x130x202xf16, #NHWC, @DDR>) -> memref<1x1x130x202xf16, #NHWC, @DDR>

    return %15 : memref<1x1x130x202xf16, #NHWC, @DDR>

    // CHECK:       [[CONST:%.*]] = const.Declare memref<1x1x1x202xf16, #NHWC, @DDR> = dense<0.000000e+00> : tensor<1x1x1x202xf16, {order = #NHWC}>

    // CHECK:       [[CONST_0:%.*]] = const.Declare memref<1x1x128x1xf16, #NHWC, @DDR> = dense<0.000000e+00> : tensor<1x1x128x1xf16, {order = #NHWC}>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x1x130x202xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 1, 0] [1, 1, 128, 202] :
    // CHECK-SAME:      memref<1x1x130x202xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[SUBVIEW_0_1:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 1] [1, 1, 128, 200] :
    // CHECK-SAME:      memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR> to
    // CHECK-SAME:      memref<1x1x128x200xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[COPY_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg1: memref<1x1x128x200xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_1]] as %arg2: memref<1x1x128x200xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:          -> memref<1x1x128x200xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR> {
    // CHECK:       [[COPY_1_INNER:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1 : memref<1x1x128x200xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs(%arg2 : memref<1x1x128x200xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:          -> memref<1x1x128x200xf16, {order = #NHWC, strides = [25856, 1, 202, 1]}, @DDR>

    // CHECK:       [[SUBVIEW_0_2:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 0] [1, 1, 128, 1] :
    // CHECK-SAME:      memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR> to
    // CHECK-SAME:      memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[COPY_RESULT_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONST_0]] : memref<1x1x128x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_2]] : memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:          -> memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[SUBVIEW_0_3:%.*]] = VPUIP.SubView [[SUBVIEW_0]] [0, 0, 0, 201] [1, 1, 128, 1] :
    // CHECK-SAME:      memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR> to
    // CHECK-SAME:      memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[COPY_RESULT_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONST_0]] : memref<1x1x128x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_3]] : memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:          -> memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[CONCAT_PARENT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_1]], [[COPY_RESULT_0]], [[COPY_RESULT_1]] :
    // CHECK-SAME:          memref<1x1x128x200xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>,
    // CHECK-SAME:          memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>,
    // CHECK-SAME:          memref<1x1x128x1xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0]] : memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:          -> memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[SUBVIEW_1_2:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 1, 1, 202] :
    // CHECK-SAME:      memref<1x1x130x202xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[COPY_RESULT_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONST]] : memref<1x1x1x202xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_1_2]] : memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:          -> memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[SUBVIEW_1_3:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 129, 0] [1, 1, 1, 202] :
    // CHECK-SAME:      memref<1x1x130x202xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[COPY_RESULT_3:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONST]] : memref<1x1x1x202xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_1_3]] : memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:          -> memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>

    // CHECK:       [[CONCAT_CHILD:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[CONCAT_PARENT]], [[COPY_RESULT_2]], [[COPY_RESULT_3]] :
    // CHECK-SAME:          memref<1x1x128x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>,
    // CHECK-SAME:          memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>,
    // CHECK-SAME:          memref<1x1x1x202xf16, {order = #NHWC, strides = [26260, 1, 202, 1]}, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_0]] : memref<1x1x130x202xf16, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x1x130x202xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCAT_CHILD]] : memref<1x1x130x202xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.99909667968750004:124>

func.func @DDR2DDROfConcatInputStrideCopy(%arg0: memref<1x512x9x32x!qElemType, #NHWC, @DDR>) -> memref<1x512x18x33x!qElemType, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x512x18x32x!qElemType, #NHWC, @DDR>

    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 512, 9, 32] [1, 1, 2, 1] : memref<1x512x18x32x!qElemType, #NHWC, @DDR> to memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>
    %2 = VPUIP.Copy inputs(%arg0 : memref<1x512x9x32x!qElemType, #NHWC, @DDR>) outputs(%1 : memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>) -> memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 0, 1, 0] [1, 512, 9, 32] [1, 1, 2, 1] : memref<1x512x18x32x!qElemType, #NHWC, @DDR> to memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>
    %4 = VPUIP.Copy inputs(%arg0 : memref<1x512x9x32x!qElemType, #NHWC, @DDR>) outputs(%3 : memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>) -> memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>

    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>, memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>) outputs(%0 : memref<1x512x18x32x!qElemType, #NHWC, @DDR>) -> memref<1x512x18x32x!qElemType, #NHWC, @DDR>

    %6 = VPUIP.SubView %5 [0, 0, 0, 31] [1, 512, 18, 1] : memref<1x512x18x32x!qElemType, #NHWC, @DDR> to memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>

    %7 = memref.alloc() : memref<1x512x18x33x!qElemType, #NHWC, @DDR>

    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 512, 18, 32] : memref<1x512x18x33x!qElemType, #NHWC, @DDR> to memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>
    %9 = VPUIP.Copy inputs(%5 : memref<1x512x18x32x!qElemType, #NHWC, @DDR>) outputs(%8 : memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %10 = VPUIP.SubView %7 [0, 0, 0, 32] [1, 512, 18, 1] : memref<1x512x18x33x!qElemType, #NHWC, @DDR> to memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>
    %11 = VPUIP.Copy inputs(%6 : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>) outputs(%10 : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) -> memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    %12 = VPUIP.ConcatView inputs(%9, %11 : memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>, memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>) outputs(%7 : memref<1x512x18x33x!qElemType, #NHWC, @DDR>) -> memref<1x512x18x33x!qElemType, #NHWC, @DDR>
    return %12 : memref<1x512x18x33x!qElemType, #NHWC, @DDR>

    // CHECK:       [[STRID_BUFF_1:%.*]] = memref.alloc() : memref<1x512x18x32x!qElemType, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_SB_10:%.*]] = VPUIP.SubView [[STRID_BUFF_1:%.*]] [0, 0, 0, 0] [1, 512, 9, 32] [1, 1, 2, 1] :
    // CHECK-SAME:      memref<1x512x18x32x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>

    // CHECK:       [[COPY_SB_10:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x512x9x32x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_SB_10]] : memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_SB_11:%.*]] = VPUIP.SubView [[STRID_BUFF_1:%.*]] [0, 0, 1, 0] [1, 512, 9, 32] [1, 1, 2, 1] :
    // CHECK-SAME:      memref<1x512x18x32x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>

    // CHECK:       [[COPY_SB_11:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x512x9x32x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_SB_11]] : memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>

    // CHECK:       [[CONCAT_SB_1:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_SB_10]], [[COPY_SB_11]] :
    // CHECK-SAME:          memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>,
    // CHECK-SAME:          memref<1x512x9x32x!qElemType, {order = #NHWC, strides = [294912, 1, 32768, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[STRID_BUFF_1:%.*]] : memref<1x512x18x32x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x32x!qElemType, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_FOR_COPY_0:%.*]] = VPUIP.SubView [[CONCAT_SB_1]] [0, 0, 0, 31] [1, 512, 18, 1] :
    // CHECK-SAME:      memref<1x512x18x32x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x512x18x33x!qElemType, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0_1:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 512, 18, 32] :
    // CHECK-SAME:      memref<1x512x18x33x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_FROM_CONCAT_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONCAT_SB_1]] : memref<1x512x18x32x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_1]] : memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    // CHECK:       [[SUBVIEW_0_2:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 32] [1, 512, 18, 1] :
    // CHECK-SAME:      memref<1x512x18x33x!qElemType, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    // CHECK:       [[COPY_FROM_SUBVIEW:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_FOR_COPY_0]] : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [294912, 1, 16384, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0_2]] : memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>

    // CHECK:       [[CONCAT_PARENT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_FROM_CONCAT_0]], [[COPY_FROM_SUBVIEW]] :
    // CHECK-SAME:          memref<1x512x18x32x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>,
    // CHECK-SAME:          memref<1x512x18x1x!qElemType, {order = #NHWC, strides = [304128, 1, 16896, 512]}, @DDR>)
    // CHECK-SAME:      outputs([[BUFF_0]] : memref<1x512x18x33x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x512x18x33x!qElemType, #NHWC, @DDR>

    // CHECK:       return [[CONCAT_PARENT]] : memref<1x512x18x33x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>

func.func @ParallelDDR2DDRCopyOutputWithSlice(%in0 : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>,
                                %in1 : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>)
                                    -> memref<1x512x10x17x!qElemType, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>
    %1 = memref.alloc() : memref<1x512x9x17x!qElemType, #NHWC, @DDR>
    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg2: memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>)
                                outputs(%1 as %arg3: memref<1x512x9x17x!qElemType, #NHWC>)
                                    -> memref<1x512x9x17x!qElemType, #NHWC, @DDR> {
        %1132 = VPUIP.Copy inputs(%arg2 : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>)
                           outputs(%arg3 : memref<1x512x9x17x!qElemType, #NHWC>)
                               -> memref<1x512x9x17x!qElemType, #NHWC>
    }

  %3 = VPUIP.SubView %2 [0, 0, 8, 0] [1, 512, 1, 17] : memref<1x512x9x17x!qElemType, #NHWC, @DDR> to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>

  %4 = memref.alloc() : memref<1x512x1x17x!qElemType, #NHWC, @DDR>

  %5 = VPUIP.Copy inputs(%3 : memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>) outputs(%4 : memref<1x512x1x17x!qElemType, #NHWC, @DDR>)
  -> memref<1x512x1x17x!qElemType, #NHWC, @DDR>

  %6 = memref.alloc() : memref<1x512x10x17x!qElemType, #NHWC, @DDR>

  %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 512, 9, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
  to memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

  %8 = VPUIP.Copy inputs(%2 : memref<1x512x9x17x!qElemType, #NHWC, @DDR>) outputs(%7 : memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>)
  -> memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

  %9 = VPUIP.SubView %6 [0, 0, 9, 0] [1, 512, 1, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
  to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

  %10 = VPUIP.Copy inputs(%5 : memref<1x512x1x17x!qElemType, #NHWC, @DDR>) outputs(%9 : memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>)
  -> memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

  %11 = VPUIP.ConcatView inputs(%8, %10 : memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) outputs(%6 : memref<1x512x10x17x!qElemType, #NHWC, @DDR>) -> memref<1x512x10x17x!qElemType, #NHWC, @DDR>

    return %11 : memref<1x512x10x17x!qElemType, #NHWC, @DDR>

    // CHECK:   [[BUFF_0:%.*]] = memref.alloc() : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>
    // CHECK:   [[BUFF_1:%.*]] = memref.alloc() : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 512, 9, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR> to memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>
    // CHECK:   [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs([[BUFF_0]] as %arg2: memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>) outputs([[SUBVIEW_0]] as %arg3: memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) -> memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR> {
    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 0, 9, 0] [1, 512, 1, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR> to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>
    // CHECK:   [[SUBVIEW_2:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 8, 0] [1, 512, 1, 17] : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN> to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>
    // CHECK:   [[COPY_1:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_2]] as %arg2: memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>) outputs([[SUBVIEW_1]] as %arg3: memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) -> memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR> {
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) outputs([[BUFF_1]] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>) -> memref<1x512x10x17x!qElemType, #NHWC, @DDR>
    // CHECK:   return [[CONCAT]] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>

func.func @ParallelDDR2DDRCopyOutputWithSubview(%in0 : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>,
                                %in1 : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>)
                                    -> memref<1x512x10x17x!qElemType, #NHWC, @DDR> {
    %0 = memref.alloc() : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>
    %1 = memref.alloc() : memref<1x512x9x17x!qElemType, #NHWC, @DDR>
    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg2: memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>)
                                outputs(%1 as %arg3: memref<1x512x9x17x!qElemType, #NHWC>)
                                    -> memref<1x512x9x17x!qElemType, #NHWC, @DDR> {
        %932 = VPUIP.Copy inputs(%arg2 : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>)
                           outputs(%arg3 : memref<1x512x9x17x!qElemType, #NHWC>)
                               -> memref<1x512x9x17x!qElemType, #NHWC>
    }

  %3 = VPUIP.SubView %2 [0, 0, 8, 0] [1, 512, 1, 17] : memref<1x512x9x17x!qElemType, #NHWC, @DDR> to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>

  %4 = memref.alloc() : memref<1x512x10x17x!qElemType, #NHWC, @DDR>

  %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 512, 9, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
  to memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

  %6 = VPUIP.Copy inputs(%2 : memref<1x512x9x17x!qElemType, #NHWC, @DDR>) outputs(%5 : memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>)
  -> memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

  %7 = VPUIP.SubView %4 [0, 0, 9, 0] [1, 512, 1, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
  to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

  %8 = VPUIP.Copy inputs(%3 : memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @DDR>) outputs(%7 : memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>)
  -> memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>

  %9 = VPUIP.ConcatView inputs(%6, %8 : memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) outputs(%4 : memref<1x512x10x17x!qElemType, #NHWC, @DDR>) -> memref<1x512x10x17x!qElemType, #NHWC, @DDR>

    return %9 : memref<1x512x10x17x!qElemType, #NHWC, @DDR>

    // CHECK:   [[BUFF_0:%.*]] = memref.alloc() : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>
    // CHECK:   [[BUFF_1:%.*]] = memref.alloc() : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 512, 9, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR> to memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>
    // CHECK:   [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs([[BUFF_0]] as %arg2: memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN>) outputs([[SUBVIEW_0]] as %arg3: memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) -> memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR> {
    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 0, 9, 0] [1, 512, 1, 17] : memref<1x512x10x17x!qElemType, #NHWC, @DDR> to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>
    // CHECK:   [[SUBVIEW_2:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 8, 0] [1, 512, 1, 17] : memref<1x512x9x17x!qElemType, #NHWC, @CMX_NN> to memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>
    // CHECK:   [[COPY_1:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_2]] as %arg2: memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [78336, 1, 8704, 512]}, @CMX_NN>) outputs([[SUBVIEW_1]] as %arg3: memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) -> memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR> {
    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : memref<1x512x9x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>, memref<1x512x1x17x!qElemType, {order = #NHWC, strides = [87040, 1, 8704, 512]}, @DDR>) outputs([[BUFF_1]] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>) -> memref<1x512x10x17x!qElemType, #NHWC, @DDR>
    // CHECK:   return [[CONCAT]] : memref<1x512x10x17x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DuplicatedType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

func.func @NCEClusterCopyOpSequence() -> !DuplicatedType {
    %0 = VPURT.AllocDistributed -> !DuplicatedType
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // spill to DDR
    %2 = VPUIP.NCEClusterTiling
            inputs(%0 as %arg62: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
            outputs(%1 as %arg63: memref<1x144x64x128xf16, #NHWC>)
                -> memref<1x144x64x128xf16, #NHWC, @DDR> {
        %inner = VPUIP.Copy
                inputs(%arg62 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                outputs(%arg63 : memref<1x144x64x128xf16, #NHWC>)
                    -> memref<1x144x64x128xf16, #NHWC>
    }

    %3 = VPURT.AllocDistributed -> !DuplicatedType

    // read to NN_CMX
    %4 = VPUIP.NCEClusterTiling
            inputs(%2 as %arg62: memref<1x144x64x128xf16, #NHWC>)
            outputs(%3 as %arg63: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                -> !DuplicatedType {
        %inner = VPUIP.Copy
                inputs(%arg62 : memref<1x144x64x128xf16, #NHWC>)
                outputs(%arg63 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    }

    return %4 : !DuplicatedType

    // CHECK:       [[BUFFER:%.*]] = VPURT.AllocDistributed
    // CHECK-NOT:   memref.alloc()
    // CHECK-NOT:   VPUIP.NCEClusterTiling
    // CHECK-NOT:   VPURT.AllocDistributed
    // CHECK-NOT:   VPUIP.NCEClusterTiling
    // CHECK:       return [[BUFFER]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DuplicatedType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

!CompatibleType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

func.func @NCEClusterCopyOpSequenceWithCast() -> !CompatibleType {
    %0 = VPURT.AllocDistributed -> !DuplicatedType
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // spill to DDR
    %2 = VPUIP.NCEClusterTiling
            inputs(%0 as %arg62: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
            outputs(%1 as %arg63: memref<1x144x64x128xf16, #NHWC>)
                -> memref<1x144x64x128xf16, #NHWC, @DDR> {
        %inner = VPUIP.Copy
                inputs(%arg62 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                outputs(%arg63 : memref<1x144x64x128xf16, #NHWC>)
                    -> memref<1x144x64x128xf16, #NHWC>
    }

    %3 = VPURT.AllocDistributed -> !CompatibleType

    // read to NN_CMX
    %4 = VPUIP.NCEClusterTiling
            inputs(%2 as %arg62: memref<1x144x64x128xf16, #NHWC>)
            outputs(%3 as %arg63: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                -> !CompatibleType {
        %inner = VPUIP.Copy
                inputs(%arg62 : memref<1x144x64x128xf16, #NHWC>)
                outputs(%arg63 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    }

    return %4 : !CompatibleType

    // CHECK:       [[BUFFER:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK-NOT:   memref.alloc()
    // CHECK-NOT:   VPUIP.NCEClusterTiling
    // CHECK-NOT:   VPURT.AllocDistributed
    // CHECK-NOT:   VPUIP.NCEClusterTiling
    // CHECK:       [[CAST:%.*]] = VPUIP.DistributedCast inputs([[BUFFER]]
    // CHECK:       return [[CAST]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DuplicatedType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

!DuplicatedDDRType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @DDR, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

func.func @NCEClusterCopyOpSequenceNoChange() -> !DuplicatedDDRType {
    %0 = VPURT.AllocDistributed -> !DuplicatedType
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // spill to DDR
    %2 = VPUIP.NCEClusterTiling
            inputs(%0 as %arg62: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
            outputs(%1 as %arg63: memref<1x144x64x128xf16, #NHWC>)
                -> memref<1x144x64x128xf16, #NHWC, @DDR> {
        %inner = VPUIP.Copy
                inputs(%arg62 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                outputs(%arg63 : memref<1x144x64x128xf16, #NHWC>)
                    -> memref<1x144x64x128xf16, #NHWC>
    }

    %3 = VPURT.AllocDistributed -> !DuplicatedDDRType

    // DDR to DDR
    %4 = VPUIP.NCEClusterTiling
            inputs(%2 as %arg62: memref<1x144x64x128xf16, #NHWC>)
            outputs(%3 as %arg63: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                -> !DuplicatedDDRType {
        %inner = VPUIP.Copy
                inputs(%arg62 : memref<1x144x64x128xf16, #NHWC>)
                outputs(%arg63 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    }

    return %4 : !DuplicatedDDRType

    // CHECK:       VPURT.AllocDistributed
    // CHECK:       memref.alloc()
    // CHECK:       VPUIP.NCEClusterTiling
    // CHECK:       VPURT.AllocDistributed
    // CHECK:       [[COPY1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK:       return [[COPY1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DuplicatedType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

func.func @FuseDDRCopyToTheFrontOfTillingCopy() -> memref<1x144x64x128xf16, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !DuplicatedType
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // spill to DDR
    %2 = VPUIP.NCEClusterTiling
            inputs(%0 as %arg62: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
            outputs(%1 as %arg63: memref<1x144x64x128xf16, #NHWC>)
                -> memref<1x144x64x128xf16, #NHWC, @DDR> {
        %inner = VPUIP.Copy
                inputs(%arg62 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                outputs(%arg63 : memref<1x144x64x128xf16, #NHWC>)
                    -> memref<1x144x64x128xf16, #NHWC>
    }

    %3 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // DDR to DDR
    %4 = VPUIP.Copy
                inputs(%2 : memref<1x144x64x128xf16, #NHWC, @DDR>)
                outputs(%3 : memref<1x144x64x128xf16, #NHWC, @DDR>)
                    -> memref<1x144x64x128xf16, #NHWC, @DDR>

    return %4 : memref<1x144x64x128xf16, #NHWC, @DDR>

    // CHECK:       [[BUF0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[BUF1:%.*]] = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>
    // CHECK:       [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs([[BUF0]] as %arg0: memref<1x144x64x128xf16, #NHWC, @CMX_NN>) outputs([[BUF1]] as %arg1: memref<1x144x64x128xf16, #NHWC, @DDR>) -> memref<1x144x64x128xf16, #NHWC, @DDR> {
    // CHECK:           VPUIP.Copy inputs(%arg0 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x144x64x128xf16, #NHWC, @DDR>) -> memref<1x144x64x128xf16, #NHWC, @DDR>
    // CHECK:       }
    // CHECK:       return [[COPY0]] : memref<1x144x64x128xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x3x12x12xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!InputStub_CMX = memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>
!SpilledOutput_DDR = memref<1x3x12x12xf16, #NHWC, @DDR>

func.func @FuseCMXCopyToTheFrontOfTillingCopy() -> !InputStub_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = memref.alloc() : !SpilledOutput_DDR
  %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x3x12x12xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg1: !SpilledOutput_DDR) -> !SpilledOutput_DDR {
      VPUIP.Copy inputs(%arg0: memref<1x3x12x12xf16, #NHWC, @CMX_NN>) outputs(%arg1: !SpilledOutput_DDR) -> !SpilledOutput_DDR
  }

  %3 = memref.alloc() : !InputStub_CMX
  %4 = VPUIP.Copy inputs(%2 : !SpilledOutput_DDR) outputs(%3 : !InputStub_CMX) -> !InputStub_CMX

  return %4 : !InputStub_CMX
  // CHECK:  [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x12x12xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
  // CHECK:  [[BUF_1:%.*]] = memref.alloc() : memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:  [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs([[BUF_0]] as %arg0: memref<1x3x12x12xf16, #NHWC, @CMX_NN>) outputs([[BUF_1]] as %arg1: memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]> {
  // CHECK:              VPUIP.Copy inputs(%arg0 : memref<1x3x12x12xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:  }
  // CHECK:  return [[COPY_0]] : memref<1x3x12x12xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDRToCMXCopyWithConcatViewWithCopy(%arg0: memref<1x3x128x128xf16, #NHWC, @DDR>)
                                    -> memref<1x16x128x128xf16, #NHWC, @CMX_NN> {
  %cst = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]
  %0 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 3, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

  %3 = VPUIP.SubView %0 [0, 3, 0, 0] [1, 13, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %4 = VPUIP.Copy inputs(%cst : memref<1x13x128x128xf16, #NHWC>)
                outputs(%3 : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

  %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
                        outputs(%0 : memref<1x16x128x128xf16, #NHWC, @DDR>) -> memref<1x16x128x128xf16, #NHWC, @DDR>

  %6 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>
  %7 = VPUIP.Copy inputs(%5 : memref<1x16x128x128xf16, #NHWC, @DDR>)
                  outputs(%6 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    return %7 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]

    // CHECK:   [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 3, 128, 128]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @CMX_NN> to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>
    // CHECK:   [[COPY_0:%.*]] =  VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
    // CHECK:                                outputs([[SUBVIEW_0]] : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 3, 0, 0] [1, 13, 128, 128]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @CMX_NN> to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>
    // CHECK:   [[COPY_1:%.*]] =  VPUIP.Copy inputs([[CST]] : memref<1x13x128x128xf16, #NHWC>)
    // CHECK:                                outputs([[SUBVIEW_1]] : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>,
    // CHECK:       memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>)
    // CHECK:       outputs([[OUT_BUFF]] :  memref<1x16x128x128xf16, #NHWC, @CMX_NN>)

    // CHECK:   return [[CONCAT]] :  memref<1x16x128x128xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDRToCMXCopyWithConcatViewWithCopylastCopiesWithSubview(%arg0: memref<1x8x128x128xf16, #NHWC, @DDR>)
                                    -> memref<1x16x128x128xf16, #NHWC, @CMX_NN> {
  %cst = const.Declare memref<1x8x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<131072xf16>, [#const.Reshape<[1, 8, 128, 128]>, #const.Reorder<#NHWC>]
  %0 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 8, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<1x8x128x128xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

  %3 = VPUIP.SubView %0 [0, 8, 0, 0] [1, 8, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %4 = VPUIP.Copy inputs(%cst : memref<1x8x128x128xf16, #NHWC>)
                outputs(%3 : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

  %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
                        outputs(%0 : memref<1x16x128x128xf16, #NHWC, @DDR>) -> memref<1x16x128x128xf16, #NHWC, @DDR>

  %6 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>

  %11 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 8, 128, 128] [1, 2, 1, 1]: memref<1x16x128x128xf16, #NHWC, @CMX_NN>
        to memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>

  %7 = VPUIP.Copy inputs(%5 : memref<1x16x128x128xf16, #NHWC, @DDR>)
                  outputs(%11 : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>) -> memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>

  %12 = VPUIP.SubView %6 [0, 8, 0, 0] [1, 8, 128, 128] [1, 2, 1, 1] : memref<1x16x128x128xf16, #NHWC, @CMX_NN>
        to memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>

  %8 = VPUIP.Copy inputs(%5 : memref<1x16x128x128xf16, #NHWC, @DDR>)
                  outputs(%12 : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>) -> memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>

  %9 = VPUIP.ConcatView inputs(%7, %8 : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>, memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>)
                        outputs(%6 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x16x128x128xf16, #NHWC, @CMX_NN>

  return %9 :memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<1x8x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<131072xf16>, [#const.Reshape<[1, 8, 128, 128]>, #const.Reorder<#NHWC>]

    // CHECK:   [[CONCAT_BUFF:%.*]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[CONCAT_BUFF]] [0, 0, 0, 0] [1, 8, 128, 128]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @DDR> to memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    // CHECK:   [[COPY_0:%.*]] =  VPUIP.Copy inputs(%arg0 : memref<1x8x128x128xf16, #NHWC, @DDR>)
    // CHECK:                                outputs([[SUBVIEW_0]] : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[CONCAT_BUFF]] [0, 8, 0, 0] [1, 8, 128, 128]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @DDR> to memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    // CHECK:   [[COPY_1:%.*]] =  VPUIP.Copy inputs([[CST]] : memref<1x8x128x128xf16, #NHWC>)
    // CHECK:                                outputs([[SUBVIEW_1]] : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

    // CHECK:   [[CONCAT1:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>,
    // CHECK:       memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
    // CHECK:       outputs([[CONCAT_BUFF]] :  memref<1x16x128x128xf16, #NHWC, @DDR>)

    // CHECK:   [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK:   [[SUBVIEW_3:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 8, 128, 128] [1, 2, 1, 1]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @CMX_NN> to memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>
    // CHECK:   [[COPY_3:%.*]] =  VPUIP.Copy inputs([[CONCAT1]] : memref<1x16x128x128xf16, #NHWC, @DDR>)
    // CHECK:                                outputs([[SUBVIEW_3]] : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>) -> memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>

    // CHECK:   [[SUBVIEW_4:%.*]] = VPUIP.SubView [[OUT_BUFF]]  [0, 8, 0, 0] [1, 8, 128, 128] [1, 2, 1, 1]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @CMX_NN> to memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>
    // CHECK:   [[COPY_4:%.*]] =  VPUIP.Copy inputs([[CONCAT1]] : memref<1x16x128x128xf16, #NHWC, @DDR>)
    // CHECK:                                outputs([[SUBVIEW_4]] : memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>) -> memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>

    // CHECK:   [[CONCAT2:%.*]] = VPUIP.ConcatView inputs([[COPY_3]], [[COPY_4]]
    // CHECK:       memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>,
    // CHECK:       memref<1x8x128x128xf16, {order = #NHWC, strides = [262144, 2, 2048, 16]}, @CMX_NN>)
    // CHECK:       outputs([[OUT_BUFF]] :  memref<1x16x128x128xf16, #NHWC, @CMX_NN>)

    // CHECK:   return [[CONCAT2]] :  memref<1x16x128x128xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDRToCMXCopyWithConcatViewWithMultiCopy(%arg0: memref<1x3x128x128xf16, #NHWC, @DDR>)
                                    -> (memref<1x16x128x128xf16, #NHWC, @CMX_NN>,memref<1x16x128x128xf16, #NHWC, @CMX_NN>) {
  %cst = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]
  %0 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 3, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

  %3 = VPUIP.SubView %0 [0, 3, 0, 0] [1, 13, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %4 = VPUIP.Copy inputs(%cst : memref<1x13x128x128xf16, #NHWC>)
                outputs(%3 : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

  %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
                        outputs(%0 : memref<1x16x128x128xf16, #NHWC, @DDR>) -> memref<1x16x128x128xf16, #NHWC, @DDR>

  %6 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>
  %7 = VPUIP.Copy inputs(%5 : memref<1x16x128x128xf16, #NHWC, @DDR>)
                  outputs(%6 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x16x128x128xf16, #NHWC, @CMX_NN>

  %8 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>
  %9 = VPUIP.Copy inputs(%5 : memref<1x16x128x128xf16, #NHWC, @DDR>)
                  outputs(%8 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x16x128x128xf16, #NHWC, @CMX_NN>

  return %7, %9 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>, memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]

    // CHECK:   [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @CMX_NN>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 3, 128, 128]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @CMX_NN> to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>
    // CHECK:   [[COPY_0:%.*]] =  VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
    // CHECK:                                outputs([[SUBVIEW_0]] : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 3, 0, 0] [1, 13, 128, 128]
    // CHECK:       memref<1x16x128x128xf16, #NHWC, @CMX_NN> to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>
    // CHECK:   [[COPY_1:%.*]] =  VPUIP.Copy inputs([[CST]] : memref<1x13x128x128xf16, #NHWC>)
    // CHECK:                                outputs([[SUBVIEW_1]] : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>,
    // CHECK:       memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>)
    // CHECK:       outputs([[OUT_BUFF]] :  memref<1x16x128x128xf16, #NHWC, @CMX_NN>)

    // CHECK:   return [[CONCAT]], [[CONCAT]] :  memref<1x16x128x128xf16, #NHWC, @CMX_NN>, memref<1x16x128x128xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

func.func @DDRToCMXCopyWithConcatViewWithClusterCopy(%arg0: memref<1x3x128x128xf16, #NHWC, @DDR>)
                                    -> !OutputDistributed {
  %cst = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]
  %0 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 3, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

  %3 = VPUIP.SubView %0 [0, 3, 0, 0] [1, 13, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
  %4 = VPUIP.Copy inputs(%cst : memref<1x13x128x128xf16, #NHWC>)
                outputs(%3 : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

  %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
                        outputs(%0 : memref<1x16x128x128xf16, #NHWC, @DDR>) -> memref<1x16x128x128xf16, #NHWC, @DDR>

  %6 = VPURT.AllocDistributed -> !OutputDistributed
  %7 = VPUIP.NCEClusterTiling inputs(%5 as %arg2: memref<1x16x128x128xf16, #NHWC>) outputs(%6 as %arg3: memref<1x16x128x128xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
    %8 = VPUIP.Copy inputs(%arg2 : memref<1x16x128x128xf16, #NHWC>) outputs(%arg3 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x16x128x128xf16, #NHWC, @CMX_NN>
  }

    return %7 : !OutputDistributed

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]

    // CHECK:   [[OUT_BUFF:%.*]] = VPURT.AllocDistributed
    // CHECK:       -> !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 3, 128, 128]
    // CHECK:       !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       !VPUIP.DistributedBuffer<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[COPY_0:%.*]] =  VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x3x128x128xf16, #NHWC, @DDR>)
    // CHECK:       outputs([[SUBVIEW_0]] as %arg2: memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>)
    // CHECK:       -> !VPUIP.DistributedBuffer<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 3, 0, 0] [1, 13, 128, 128]
    // CHECK:       !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK"       !VPUIP.DistributedBuffer<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[COPY_1:%.*]] =  VPUIP.NCEClusterTiling inputs([[CST]] as %arg1: memref<1x13x128x128xf16, #NHWC>)
    // CHECK:       outputs([[SUBVIEW_1]] as %arg2: memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN>)
    // CHECK:       -> !VPUIP.DistributedBuffer<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       !VPUIP.DistributedBuffer<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       !VPUIP.DistributedBuffer<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:       outputs([[OUT_BUFF]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:   return [[CONCAT]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    80x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

func.func @DDRToCMXCopyWithConcatViewNoChange(%arg0: memref<76x64x1x1xf16, #NHWC, @DDR>)
                                    -> !OutputDistributed {
  %cst = const.Declare memref<4x64x1x1xf16, #NHWC> = dense<0.000000e+00> : tensor<22800xf16>, [#const.SubView<[0], [256]>, #const.Reshape<[4, 64, 1, 1]>, #const.Reorder<#NHWC>]
  %0 = memref.alloc() : memref<80x64x1x1xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [76, 64, 1, 1] : memref<80x64x1x1xf16, #NHWC, @DDR>
        to memref<76x64x1x1xf16, #NHWC, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<76x64x1x1xf16, #NHWC, @DDR>)
                  outputs(%1 : memref<76x64x1x1xf16, #NHWC, @DDR>) -> memref<76x64x1x1xf16, #NHWC, @DDR>

  %3 = VPUIP.SubView %0 [76, 0, 0, 0] [4, 64, 1, 1] : memref<80x64x1x1xf16, #NHWC, @DDR>
        to memref<4x64x1x1xf16, #NHWC, @DDR>
  %4 = VPUIP.Copy inputs(%cst : memref<4x64x1x1xf16, #NHWC>)
                outputs(%3 : memref<4x64x1x1xf16, #NHWC, @DDR>) -> memref<4x64x1x1xf16, #NHWC, @DDR>

  %5 = VPUIP.ConcatView inputs(%2, %4 : memref<76x64x1x1xf16, #NHWC, @DDR>, memref<4x64x1x1xf16, #NHWC, @DDR>)
                        outputs(%0 : memref<80x64x1x1xf16, #NHWC, @DDR>) -> memref<80x64x1x1xf16, #NHWC, @DDR>

  %6 = VPURT.AllocDistributed -> !OutputDistributed
  %7 = VPUIP.NCEClusterTiling inputs(%5 as %arg2: memref<80x64x1x1xf16, #NHWC>) outputs(%6 as %arg3: memref<80x64x1x1xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
    %8 = VPUIP.Copy inputs(%arg2 : memref<80x64x1x1xf16, #NHWC>) outputs(%arg3 : memref<80x64x1x1xf16, #NHWC, @CMX_NN>) -> memref<80x64x1x1xf16, #NHWC, @CMX_NN>
  }

    return %7 : !OutputDistributed

    // CHECK-DAG:   [[CST:%.*]] = const.Declare memref<4x64x1x1xf16, #NHWC> = dense<0.000000e+00> : tensor<22800xf16>, [#const.SubView<[0], [256]>, #const.Reshape<[4, 64, 1, 1]>, #const.Reorder<#NHWC>]

    // CHECK:   [[OUT_BUFF:%.*]] = memref.alloc() : memref<80x64x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [76, 64, 1, 1] : memref<80x64x1x1xf16, #NHWC, @DDR> to memref<76x64x1x1xf16, #NHWC, @DDR>
    // CHECK:   [[COPY_0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<76x64x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_0]] : memref<76x64x1x1xf16, #NHWC, @DDR>) -> memref<76x64x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [76, 0, 0, 0] [4, 64, 1, 1] : memref<80x64x1x1xf16, #NHWC, @DDR> to memref<4x64x1x1xf16, #NHWC, @DDR>
    // CHECK:   [[COPY_1:%.*]] = VPUIP.Copy inputs(%cst : memref<4x64x1x1xf16, #NHWC>) outputs([[SUBVIEW_1]] : memref<4x64x1x1xf16, #NHWC, @DDR>) -> memref<4x64x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:       memref<76x64x1x1xf16, #NHWC, @DDR>,
    // CHECK:       memref<4x64x1x1xf16, #NHWC, @DDR>)
    // CHECK:       outputs([[OUT_BUFF]] : memref<80x64x1x1xf16, #NHWC, @DDR>)

    // CHECK:   [[CLUSTER_COPY:%.*]] = VPUIP.NCEClusterTiling

    // CHECK:   return [[CLUSTER_COPY]] : !VPUIP.DistributedBuffer<80x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DDRToCMXCopyWithConcatViewWithCopyStaticStrides(%arg0: memref<1x64x3x10xf16, #NHWC, @DDR>)
                                    -> memref<1x64x6x10xf16, #NHWC, @CMX_NN> {
  %0 = memref.alloc() : memref<1x64x6x10xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 64, 3, 10] [1, 1, 2, 1]: memref<1x64x6x10xf16, #NHWC, @DDR>
        to memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>

  %3 = VPUIP.SubView %0 [0, 0, 1, 0] [1, 64, 3, 10] [1, 1, 2, 1]: memref<1x64x6x10xf16, #NHWC, @DDR>
        to memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>
  %4 = VPUIP.Copy inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
                outputs(%3 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>

  %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>, memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>)
                        outputs(%0 : memref<1x64x6x10xf16, #NHWC, @DDR>) -> memref<1x64x6x10xf16, #NHWC, @DDR>

  %6 = memref.alloc() : memref<1x64x6x10xf16, #NHWC, @CMX_NN>
  %7 = VPUIP.Copy inputs(%5 : memref<1x64x6x10xf16, #NHWC, @DDR>)
                  outputs(%6 : memref<1x64x6x10xf16, #NHWC, @CMX_NN>) -> memref<1x64x6x10xf16, #NHWC, @CMX_NN>

    return %7 : memref<1x64x6x10xf16, #NHWC, @CMX_NN>

    // CHECK: [[OUT_BUFF:%.*]] = memref.alloc() : memref<1x64x6x10xf16, #NHWC, @CMX_NN>

    // CHECK: [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 64, 3, 10] [1, 1, 2, 1] :
    // CHECK:       memref<1x64x6x10xf16, #NHWC, @CMX_NN> to memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>
    // CHECK: [[COPY_0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
    // CHECK:       outputs([[SUBVIEW_0]] : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>

    // CHECK: [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 1, 0] [1, 64, 3, 10] [1, 1, 2, 1] :
    // CHECK:       memref<1x64x6x10xf16, #NHWC, @CMX_NN> to memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>
    // CHECK: [[COPY_1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
    // CHECK:       outputs([[SUBVIEW_1]] : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>

    // CHECK: [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>,
    // CHECK:       memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>) outputs([[OUT_BUFF]] : memref<1x64x6x10xf16, #NHWC, @CMX_NN>) -> memref<1x64x6x10xf16, #NHWC, @CMX_NN>
    // CHECK: return [[CONCAT]] : memref<1x64x6x10xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x64x6x10xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

func.func @DDRToCMXCopyWithConcatViewWithClusterCopyStaticStrides(%arg0: memref<1x64x3x10xf16, #NHWC, @DDR>)
                                    -> !OutputDistributed {
  %0 = memref.alloc() : memref<1x64x6x10xf16, #NHWC, @DDR>

  %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 64, 3, 10] [1, 1, 2, 1] : memref<1x64x6x10xf16, #NHWC, @DDR>
        to memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>
  %2 = VPUIP.Copy inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>

  %3 = VPUIP.SubView %0 [0, 0, 1, 0] [1, 64, 3, 10] [1, 1, 2, 1] : memref<1x64x6x10xf16, #NHWC, @DDR>
        to memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>
  %4 = VPUIP.Copy inputs(%arg0 : memref<1x64x3x10xf16, #NHWC, @DDR>)
                outputs(%3 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>

  %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>, memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @DDR>)
                        outputs(%0 : memref<1x64x6x10xf16, #NHWC, @DDR>) -> memref<1x64x6x10xf16, #NHWC, @DDR>

  %6 = VPURT.AllocDistributed -> !OutputDistributed
  %7 = VPUIP.NCEClusterTiling inputs(%5 as %arg2: memref<1x64x6x10xf16, #NHWC>) outputs(%6 as %arg3: memref<1x64x6x10xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
  %8 = VPUIP.Copy inputs(%arg2 : memref<1x64x6x10xf16, #NHWC>) outputs(%arg3 : memref<1x64x6x10xf16, #NHWC, @CMX_NN>) -> memref<1x64x6x10xf16, #NHWC, @CMX_NN>
  }

    return %7 : !OutputDistributed

    // CHECK: [[OUT_BUFF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x6x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 64, 3, 10] [1, 1, 2, 1] :
    // CHECK:       !VPUIP.DistributedBuffer<1x64x6x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to
    // CHECK:       !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x64x3x10xf16, #NHWC, @DDR>)
    // CHECK:       outputs([[SUBVIEW_0]] as %arg2: memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>) ->
    // CHECK:       !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:       %6 = VPUIP.Copy inputs(%arg1 : memref<1x64x3x10xf16, #NHWC, @DDR>)
    // CHECK:       outputs(%arg2 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>
    // CHECK: }

    // CHECK: [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 1, 0] [1, 64, 3, 10] [1, 1, 2, 1] :
    // CHECK:       !VPUIP.DistributedBuffer<1x64x6x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to
    // CHECK:       !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: [[COPY_1:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x64x3x10xf16, #NHWC, @DDR>)
    // CHECK:       outputs([[SUBVIEW_1]] as %arg2: memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>) ->
    // CHECK:       !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:       %6 = VPUIP.Copy inputs(%arg1 : memref<1x64x3x10xf16, #NHWC, @DDR>)
    // CHECK:       outputs(%arg2 : memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>) -> memref<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN>
    // CHECK: }

    // CHECK: [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] :
    // CHECK:       !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK:       !VPUIP.DistributedBuffer<1x64x3x10xf16, {order = #NHWC, strides = [3840, 1, 1280, 64]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) outputs([[OUT_BUFF]] : !VPUIP.DistributedBuffer<1x64x6x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x64x6x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: return [[CONCAT]] : !VPUIP.DistributedBuffer<1x64x6x10xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SiblingCopyOptimization(%in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>, %in1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>, %in2 : memref<128x1x1x4xsi32, @CMX_NN>) -> (memref<1x256x36x36xf16, #NHWC, @CMX_NN>, memref<1x128x36x36xf16, #NHWC, @CMX_NN>) {
    %0 = memref.alloc() : memref<1x128x36x36xf16, #NHWC, @CMX_NN>
    %1 = memref.alloc() : memref<1x128x36x36xf16, #NHWC, @CMX_NN>
    %2 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = "CONV"}
        input(%in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) weights(%in1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%in2 : memref<128x1x1x4xsi32, @CMX_NN>)
        parent_input(%in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) parent_output(%0 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%0 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) -> memref<1x128x36x36xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [35, 35, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, outStart = [0, 0, 0]}
            DPUTask {cluster_id = 1 : i64, outEnd = [35, 35, 127], mpe_mode = "CUBOID_16x16", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, outStart = [0, 0, 64]}
        } PPE : {
        PPETask "LPRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.199951171875 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64}
        }

    %3 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = "CONV"}
        input(%in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) weights(%in1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%in2 : memref<128x1x1x4xsi32, @CMX_NN>)
        parent_input(%in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) parent_output(%1 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%1 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) -> memref<1x128x36x36xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [35, 35, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, outStart = [0, 0, 0]}
            DPUTask {cluster_id = 1 : i64, outEnd = [35, 35, 127], mpe_mode = "CUBOID_16x16", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, outStart = [0, 0, 64]}
        } PPE : {
        PPETask "LPRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.199951171875 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64}
        }
    %4 = memref.alloc() : memref<1x256x36x36xf16, #NHWC, @CMX_NN>

    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 128, 36, 36] : memref<1x256x36x36xf16, #NHWC, @CMX_NN> to memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>
    %6 = VPUIP.Copy inputs(%2 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%5 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>

    %7 = VPUIP.SubView %4 [0, 128, 0, 0] [1, 128, 36, 36] : memref<1x256x36x36xf16, #NHWC, @CMX_NN> to memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>
    %8 = VPUIP.Copy inputs(%3 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%7 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>


    %9 = VPUIP.ConcatView inputs(%6, %8 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>, memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>)
            outputs(%4 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) -> memref<1x256x36x36xf16, #NHWC, @CMX_NN>
    %10 = memref.alloc() : memref<1x128x36x36xf16, #NHWC, @CMX_NN>
    %11 = VPUIP.Copy inputs(%2 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%10 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) -> memref<1x128x36x36xf16, #NHWC, @CMX_NN>
    return %9, %11 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>, memref<1x128x36x36xf16, #NHWC, @CMX_NN>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x256x36x36xf16, #NHWC, @CMX_NN>
    // CHECK:       [[SBUVIEW_0:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 128, 36, 36] : memref<1x256x36x36xf16, #NHWC, @CMX_NN> to memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>

    // CHECK:       [[NCETASK_0:%.*]] = VPUIP.NCEClusterTask {kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = "CONV"}
    // CHECK-SAME:      input(%arg0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) weights(%arg1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg2 : memref<128x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input(%arg0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) parent_output([[SBUVIEW_0]] : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>)
    // CHECK-SAME:      outputs([[SBUVIEW_0]] : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN> variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [35, 35, 63], outStart = [0, 0, 0], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}
    // CHECK:           DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [35, 35, 127], outStart = [0, 0, 64], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}
    // CHECK:           } PPE : {
    // CHECK:               PPETask "LPRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.199951171875 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64}
    // CHECK:           }

    // CHECK:       [[SBUVIEW_1:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 128, 0, 0] [1, 128, 36, 36] : memref<1x256x36x36xf16, #NHWC, @CMX_NN> to memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>
    // CHECK:       [[NCETASK_1:%.*]] = VPUIP.NCEClusterTask {kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = "CONV"}
    // CHECK-SAME:      input(%arg0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) weights(%arg1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg2 : memref<128x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input(%arg0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) parent_output([[SBUVIEW_1]] : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>)
    // CHECK-SAME:      outputs([[SBUVIEW_1]] : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN> variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [35, 35, 63], outStart = [0, 0, 0], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}
    // CHECK:           DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [35, 35, 127], outStart = [0, 0, 64], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}
    // CHECK:           } PPE : {
    // CHECK:               PPETask "LPRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.199951171875 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64}
    // CHECK:           }

    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[NCETASK_0]], [[NCETASK_1]] : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>, memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>)
    // CHECK-SAME:      outputs(%0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) -> memref<1x256x36x36xf16, #NHWC, @CMX_NN>

    // CHECK:       [[BUFF_1:%.*]] = memref.alloc() : memref<1x128x36x36xf16, #NHWC, @CMX_NN>
    // CHECK:       [[COPY2:%.*]] = VPUIP.Copy inputs([[NCETASK_0]] : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) outputs([[BUFF_1]] : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) -> memref<1x128x36x36xf16, #NHWC, @CMX_NN>

    // CHECK:       return [[CONCAT]], [[COPY2]] : memref<1x256x36x36xf16, #NHWC, @CMX_NN>, memref<1x128x36x36xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SiblingTilingCopyOptimization(%in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>, %in1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>, %in2 : memref<128x1x1x4xsi32, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, memref<1x128x36x36xf16, #NHWC, @DDR>) {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %9 = memref.alloc() : memref<1x128x36x36xf16, #NHWC, @DDR>

    %2 = VPUIP.NCEClusterTiling inputs(%in0 as %arg5: memref<1x256x36x36xf16, #NHWC, @CMX_NN>, %in1 as %arg6: memref<128x256x3x3xf16, #NHWC, @CMX_NN>, %in2 as %arg7: memref<128x1x1x4xsi32, @CMX_NN>)
        outputs(%0 as %arg8: memref<1x128x36x36xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
        %581 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = "CONV"}
        input(%arg5 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) weights(%arg6 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg7 : memref<128x1x1x4xsi32, @CMX_NN>)
        parent_input(%arg5 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) parent_output(%arg8 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%arg8 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) -> memref<1x128x36x36xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [35, 35, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, outStart = [0, 0, 0]}
            DPUTask {cluster_id = 1 : i64, outEnd = [35, 35, 127], mpe_mode = "CUBOID_16x16", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, outStart = [0, 0, 64]}
        } PPE : {
        PPETask "LPRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.199951171875 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64}
        }
    }
    %3 = VPUIP.NCEClusterTiling inputs(%in0 as %arg5: memref<1x256x36x36xf16, #NHWC, @CMX_NN>, %in1 as %arg6: memref<128x256x3x3xf16, #NHWC, @CMX_NN>, %in2 as %arg7: memref<128x1x1x4xsi32, @CMX_NN>)
        outputs(%1 as %arg8: memref<1x128x36x36xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
        %581 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = "CONV"}
        input(%arg5 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) weights(%arg6 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg7 : memref<128x1x1x4xsi32, @CMX_NN>)
        parent_input(%arg5 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) parent_output(%arg8 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%arg8 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) -> memref<1x128x36x36xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [35, 35, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, outStart = [0, 0, 0]}
            DPUTask {cluster_id = 1 : i64, outEnd = [35, 35, 127], mpe_mode = "CUBOID_16x16", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, outStart = [0, 0, 64]}
        } PPE : {
        PPETask "LPRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.199951171875 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64}
        }
    }

    %4 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %6 = VPUIP.NCEClusterTiling inputs(%2 as %arg5: memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%5 as %arg6: memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
        %581 = VPUIP.Copy inputs(%arg5 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%arg6 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>
    }

    %7 = VPUIP.SubView %4 [0, 128, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %8 = VPUIP.NCEClusterTiling inputs(%3 as %arg5: memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%7 as %arg6: memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
        %581 = VPUIP.Copy inputs(%arg5 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%arg6 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>
    }

    %10 = VPUIP.ConcatView inputs(%6, %8 : !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
            outputs(%4 : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %11 = VPUIP.NCEClusterTiling inputs(%2 as %arg5: memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%9 as %arg6: memref<1x128x36x36xf16, #NHWC>) -> memref<1x128x36x36xf16, #NHWC, @DDR> {
        %581 = VPUIP.Copy inputs(%arg5 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%arg6 : memref<1x128x36x36xf16, #NHWC>) -> memref<1x128x36x36xf16, #NHWC>}
    return %10, %11: !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, memref<1x128x36x36xf16, #NHWC, @DDR>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x128x36x36xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[SUBVIEW_0:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:       [[NCETASK_0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg3: memref<1x256x36x36xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:         %arg1 as %arg4: memref<128x256x3x3xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:         %arg2 as %arg5: memref<128x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_0]] as %arg6: memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:       [[INNER_0:%.*]]  = VPUIP.NCEClusterTask {kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = "CONV"}
    // CHECK-SAME:      input(%arg3 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME       weights(%arg4 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table(%arg5 : memref<128x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input(%arg3 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output(%arg6 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>)
    // CHECK-SAME:      outputs(%arg6 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN> variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [35, 35, 63], outStart = [0, 0, 0], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}
    // CHECK:           DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [35, 35, 127], outStart = [0, 0, 64], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}
    // CHECK:           } PPE : {
    // CHECK:           PPETask "LPRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.199951171875 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64}
    // CHECK:           }
    // CHECK:       }

    // CHECK:       [[TILING:%.*]] = VPUIP.NCEClusterTiling inputs([[NCETASK_0]] as %arg3: memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) outputs([[BUFF_0]] as %arg4: memref<1x128x36x36xf16, #NHWC, @DDR>) -> memref<1x128x36x36xf16, #NHWC, @DDR> {
    // CHECK:           [[INNER_COPY:%.*]] = VPUIP.Copy inputs(%arg3 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) outputs(%arg4 : memref<1x128x36x36xf16, #NHWC, @DDR>) -> memref<1x128x36x36xf16, #NHWC, @DDR>
    // CHECK:       }

    // CHECK:       [[SBUVIEW_1:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 128, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:       [[NCETASK_1:%.*]] VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg3: memref<1x256x36x36xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:          %arg1 as %arg4: memref<128x256x3x3xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:          %arg2 as %arg5: memref<128x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[SBUVIEW_1]] as %arg6: memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:       [[INNER_1:%.*]] = VPUIP.NCEClusterTask {kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = "CONV"} input(%arg3 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights(%arg4 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table(%arg5 : memref<128x1x1x4xsi32, @CMX_NN>) parent_input(%arg3 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output(%arg6 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) outputs(%arg6 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN> variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [35, 35, 63], outStart = [0, 0, 0], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}
    // CHECK:           DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [35, 35, 127], outStart = [0, 0, 64], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}
    // CHECK:           } PPE : {
    // CHECK:               PPETask "LPRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.199951171875 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64}
    // CHECK:           }
    // CHECK:       }

    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[NCETASK_0:%.*]], [[NCETASK_1:%.*]] : !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:      outputs([[BUFF_1]] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:       return [[CONCAT]], [[TILING]] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, memref<1x128x36x36xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SiblingTilingCopyOptimizationSameParent(%in0 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>, %in1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>, %in2 : memref<128x1x1x4xsi32, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%in0 as %arg5: memref<1x256x36x36xf16, #NHWC, @CMX_NN>, %in1 as %arg6: memref<128x256x3x3xf16, #NHWC, @CMX_NN>, %in2 as %arg7: memref<128x1x1x4xsi32, @CMX_NN>)
        outputs(%0 as %arg8: memref<1x128x36x36xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
        %581 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = "CONV"}
        input(%arg5 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) weights(%arg6 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg7 : memref<128x1x1x4xsi32, @CMX_NN>)
        parent_input(%arg5 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) parent_output(%arg8 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%arg8 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) -> memref<1x128x36x36xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [35, 35, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, outStart = [0, 0, 0]}
            DPUTask {cluster_id = 1 : i64, outEnd = [35, 35, 127], mpe_mode = "CUBOID_16x16", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, outStart = [0, 0, 64]}
        } PPE : {
        PPETask "LPRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.199951171875 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64}
        }
    }
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg5: memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%3 as %arg6: memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
        %581 = VPUIP.Copy inputs(%arg5 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%arg6 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>
    }

    %5 = VPUIP.SubView %2 [0, 128, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %6 = VPUIP.NCEClusterTiling inputs(%1 as %arg5: memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%5 as %arg6: memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
        %581 = VPUIP.Copy inputs(%arg5 : memref<1x128x36x36xf16, #NHWC, @CMX_NN>) outputs(%arg6 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>
    }

    %7 = VPUIP.ConcatView inputs(%4, %6 : !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
            outputs(%2 : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    return %7 : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: [[CONCAT_OUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: [[SUBVIEW_0:%.*]] = VPUIP.SubView [[CONCAT_OUT]] [0, 0, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: [[CONV:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x256x36x36xf16, #NHWC, @CMX_NN>, %arg1 as %arg4: memref<128x256x3x3xf16, #NHWC, @CMX_NN>, %arg2 as %arg5: memref<128x1x1x4xsi32, @CMX_NN>) outputs([[SUBVIEW_0]] as %arg6: memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK: [[INNER_TASK:%.*]] = VPUIP.NCEClusterTask {kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = "CONV"} input(%arg3 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) weights(%arg4 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg5 : memref<128x1x1x4xsi32, @CMX_NN>) parent_input(%arg3 : memref<1x256x36x36xf16, #NHWC, @CMX_NN>) parent_output(%arg6 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) outputs(%arg6 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN> variants : {
    // CHECK:    DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [35, 35, 63], outStart = [0, 0, 0], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}
    // CHECK:    DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [35, 35, 127], outStart = [0, 0, 64], pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}
    // CHECK:   } PPE : {
    // CHECK:    PPETask "LPRELU" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.199951171875 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64}
    // CHECK:   }
    // CHECK: }

    // CHECK: [[SUBVIEW_1:%.*]] = VPUIP.SubView [[CONCAT_OUT]] [0, 128, 0, 0] [1, 128, 36, 36] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK: [[SUBVIEW_1_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONV]] as %arg3: memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) outputs([[SUBVIEW_1]] as %arg4: memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:   [[INNER_COPY:%.*]] = VPUIP.Copy inputs(%arg3 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) outputs(%arg4 : memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>) -> memref<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN>
    // CEHCK: }

    // CHECK: [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[CONV]], [[SUBVIEW_1_COPY]] : !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x36x36xf16, {order = #NHWC, strides = [331776, 1, 9216, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) outputs([[CONCAT_OUT]] : !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x256x36x36xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:  return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.0:123>
!qElemType1 = !quant.uniform<u8:f16, 2.0:123>

!ConcatInputType = !VPUIP.DistributedBuffer<
    1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!ConvOutputType = !VPUIP.DistributedBuffer<
    1x256x26x26x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>

!ConcatOutputType = !VPUIP.DistributedBuffer<
    1x512x26x26x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>

!QCOutType = !VPUIP.DistributedBuffer<
    1x256x26x26x!qElemType1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

func.func @PropagateStridesForSiblingCopy(
                    %input0: memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>,
                    %input1: !ConcatInputType,
                    %weights: memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>
                    ) -> (memref<1x256x26x26x!qElemType, #NHWC, @DDR>, !ConcatOutputType) {
    %0 = memref.alloc() : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTiling inputs(%input0 as %arg2: memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>,
                                          %weights as %arg3: memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
                                           outputs(%0 as %arg4: memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
                                           -> !ConvOutputType {
        %11 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, minimumHardwareExecutionCost = 1081 : i64, task_type = "ELTWISE"}
            input(%arg2 : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
            weights(%arg3 : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
            parent_input(%arg2 : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
            parent_output(%arg4 : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
            outputs(%arg4 : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
                -> memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask "ADD" {clamp_high = 131 : i64, clamp_low = -124 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [26565], quant_post_shift = 0 : i64, quant_shift = [15]}
        }
    }
    %2 = VPUIP.QuantizeCast inputs(%1 : !ConvOutputType) -> memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>
    %3 = memref.alloc() : memref<1x256x26x26x!qElemType, #NHWC, @DDR>
    %4 = VPUIP.NCEClusterTiling inputs(%2 as %arg2: memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
                                outputs(%3 as %arg3: memref<1x256x26x26x!qElemType, #NHWC>)
                                -> memref<1x256x26x26x!qElemType, #NHWC, @DDR> {
        %11 = VPUIP.Copy inputs(%arg2 : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x256x26x26x!qElemType, #NHWC>) -> memref<1x256x26x26x!qElemType, #NHWC>
    }

    %5 = VPURT.AllocDistributed -> !ConcatOutputType
    %6 = VPUIP.SubView %5 [0, 0, 0, 0] [1, 256, 26, 26] : !ConcatOutputType to !ConcatInputType
    %7 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
                                outputs(%6 as %arg3: memref<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN>)
                                -> !ConcatInputType {
        %11 = VPUIP.Copy inputs(%arg2 : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN>) -> memref<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN>
    }
    %8 = VPUIP.ConcatView inputs(%7, %input1 : !ConcatInputType, !ConcatInputType) outputs(%5 : !ConcatOutputType) -> !ConcatOutputType

    return %4, %8: memref<1x256x26x26x!qElemType, #NHWC, @DDR>, !ConcatOutputType

    // Check the strides attribute is propagated to the sibling copy when a stride copy is eliminated
    // CHECK:       [[CONV_OUT_ALLOC:%.*]] =  VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x512x26x26x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[CONV_OUT_SUBVIEW:%.*]] = VPUIP.SubView %0 [0, 0, 0, 0] [1, 256, 26, 26] : !VPUIP.DistributedBuffer<1x512x26x26x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[CONV:%.*]] = VPUIP.NCEClusterTiling inputs(
    // CHECK-SAME:      %arg0 as %arg3: memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg2 as %arg4: memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[CONV_OUT_SUBVIEW]] as %arg5: memref<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:       [[QC:%.*]] = VPUIP.QuantizeCast inputs(
    // CHECK-SAME:      [[CONV]] : !VPUIP.DistributedBuffer<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:          -> memref<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN>
    // CHECK:       [[COPY_ALLOC:%.*]] = memref.alloc() : memref<1x256x26x26x!qElemType, #NHWC, @DDR>
    // CHECK:       [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs(
    // CHECK-SAME:      [[QC]] as %arg3: memref<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN>)
    // CHECK-SAME:      outputs([[COPY_ALLOC]] as %arg4: memref<1x256x26x26x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x256x26x26x!qElemType, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!DuplicatedTypeAligned = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!DuplicatedTypeUnaligned = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

func.func @NCEClusterCopyOpSequenceDUPAlign() -> !DuplicatedTypeUnaligned {
    %0 = VPURT.AllocDistributed -> !DuplicatedTypeAligned
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // spill to DDR
    %2 = VPUIP.NCEClusterTiling
            inputs(%0 as %arg62: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
            outputs(%1 as %arg63: memref<1x144x64x128xf16, #NHWC>)
                -> memref<1x144x64x128xf16, #NHWC, @DDR> {
        %inner = VPUIP.Copy
                inputs(%arg62 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                outputs(%arg63 : memref<1x144x64x128xf16, #NHWC>)
                    -> memref<1x144x64x128xf16, #NHWC>
    }

    %3 = VPURT.AllocDistributed -> !DuplicatedTypeAligned

    // read to NN_CMX
    %4 = VPUIP.NCEClusterTiling
            inputs(%2 as %arg62: memref<1x144x64x128xf16, #NHWC>)
            outputs(%3 as %arg63: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                -> !DuplicatedTypeUnaligned {
        %inner = VPUIP.Copy
                inputs(%arg62 : memref<1x144x64x128xf16, #NHWC>)
                outputs(%arg63 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    }

    return %4 : !DuplicatedTypeUnaligned

    // Don't need to check the alignment when the tensor is not split
    // The Copy ops are optimized to DistributedCast
    // CHECK:       [[BUFFER:%.*]] = VPURT.AllocDistributed
    // CHECK-NOT:   memref.alloc()
    // CHECK-NOT:   VPUIP.NCEClusterTiling
    // CHECK-NOT:   VPURT.AllocDistributed
    // CHECK-NOT:   VPUIP.NCEClusterTiling
    // CHECK:       [[CAST:%.*]] = VPUIP.DistributedCast
    // return [[CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!SegmentedTypeAligned = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!SegmentedTypeUnaligned = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

func.func @NCEClusterCopyOpSequenceSEGAlign() -> !SegmentedTypeUnaligned {
    %0 = VPURT.AllocDistributed -> !SegmentedTypeAligned
    %1 = memref.alloc() : memref<1x144x64x128xf16, #NHWC, @DDR>

    // spill to DDR
    %2 = VPUIP.NCEClusterTiling
            inputs(%0 as %arg62: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
            outputs(%1 as %arg63: memref<1x144x64x128xf16, #NHWC>)
                -> memref<1x144x64x128xf16, #NHWC, @DDR> {
        %inner = VPUIP.Copy
                inputs(%arg62 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                outputs(%arg63 : memref<1x144x64x128xf16, #NHWC>)
                    -> memref<1x144x64x128xf16, #NHWC>
    }

    %3 = VPURT.AllocDistributed -> !SegmentedTypeAligned

    // read to NN_CMX
    %4 = VPUIP.NCEClusterTiling
            inputs(%2 as %arg62: memref<1x144x64x128xf16, #NHWC>)
            outputs(%3 as %arg63: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                -> !SegmentedTypeUnaligned {
        %inner = VPUIP.Copy
                inputs(%arg62 : memref<1x144x64x128xf16, #NHWC>)
                outputs(%arg63 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    }

    return %4 : !SegmentedTypeUnaligned

    // The alignments have to be compatible for split tensors
    // The Copies are not optimized because of incompatibility
    // CHECK:       [[BUFFER:%.*]] = VPURT.AllocDistributed
    // CHECK-NOT:   VPUIP.DistributedCast
    // CHECK:       [[ALLOC:%.*]] = memref.alloc()
    // CHECK:       [[COPY:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[BUFFER]] as %arg0: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[ALLOC]] as %arg1: memref<1x144x64x128xf16, #NHWC>)
    // CHECK:           [[INNER_COPY:%.*]] = VPUIP.Copy
    // CHECK:       [[BUFFER_1:%.*]] = VPURT.AllocDistributed
    // CHECK:       [[COPY_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[COPY]] as %arg0: memref<1x144x64x128xf16, #NHWC>)
    // CHECK-SAME:      outputs([[BUFFER_1]] as %arg1: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK:           [[INNER_COPY:%.*]] = VPUIP.Copy
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!WeightsType = memref<64x64x1x1xf16, #NHWC, @DDR>
!Output_DDR = memref<32x64x1x1xf16, #NHWC, @DDR>
!OutputStub_CMX = memref<32x64x1x1xf16, #NHWC, @CMX_NN>

func.func @MoveTilingCopyBeforeSubviewForSegmentedOnN(%arg0: !WeightsType) -> (!Output_DDR, !Output_DDR) {
    %weights0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [32, 64, 1, 1] : memref<64x64x1x1xf16, #NHWC, @DDR> to memref<32x64x1x1xf16, #NHWC, @DDR>
    %weights1 = VPUIP.SubView %arg0 [32, 0, 0, 0] [32, 64, 1, 1] : memref<64x64x1x1xf16, #NHWC, @DDR> to memref<32x64x1x1xf16, #NHWC, @DDR>

    %weights0_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %weights0_copy = VPUIP.NCEClusterTiling inputs(%weights0 as %arg1: memref<32x64x1x1xf16, #NHWC>) outputs(%weights0_cmx as %arg2: memref<32x64x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
       %0 = VPUIP.Copy inputs(%arg1 : memref<32x64x1x1xf16, #NHWC>) outputs(%arg2 : memref<32x64x1x1xf16, #NHWC, @CMX_NN>) -> memref<32x64x1x1xf16, #NHWC, @CMX_NN>
    }

    %weights1_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %weights1_copy = VPUIP.NCEClusterTiling inputs(%weights1 as %arg1: memref<32x64x1x1xf16, #NHWC>) outputs(%weights1_cmx as %arg2: memref<32x64x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
       %0 = VPUIP.Copy inputs(%arg1 : memref<32x64x1x1xf16, #NHWC>) outputs(%arg2 : memref<32x64x1x1xf16, #NHWC, @CMX_NN>) -> memref<32x64x1x1xf16, #NHWC, @CMX_NN>
    }

    // simulate nce task 0
    %output0_buf  = memref.alloc() : !Output_DDR
    %output0 = VPUIP.NCEClusterTiling inputs(%weights0_copy as %arg1: !OutputStub_CMX) outputs(%output0_buf as %arg2: !Output_DDR) -> !Output_DDR {
             VPUIP.Copy inputs(%arg1: !OutputStub_CMX) outputs(%arg2: !Output_DDR) -> !Output_DDR
    }

    // simulate nce task 1
    %output1_buf  = memref.alloc() : !Output_DDR
    %output1 = VPUIP.NCEClusterTiling inputs(%weights1_copy as %arg1: !OutputStub_CMX) outputs(%output1_buf as %arg2: !Output_DDR) -> !Output_DDR {
             VPUIP.Copy inputs(%arg1: !OutputStub_CMX) outputs(%arg2: !Output_DDR) -> !Output_DDR
    }

    return %output0, %output1: !Output_DDR, !Output_DDR


    // CHECK:       [[WEIGHTS_BUF_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       [[WEIGHTS_COPY:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<64x64x1x1xf16, #NHWC, @DDR>) outputs([[WEIGHTS_BUF_CMX]] as %arg2: memref<64x64x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
    // CHECK:                                  VPUIP.Copy inputs(%arg1 : memref<64x64x1x1xf16, #NHWC, @DDR>) outputs(%arg2 : memref<64x64x1x1xf16, #NHWC, @CMX_NN>) -> memref<64x64x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:       }
    // CHECK:       [[SUBVIEW0:%.*]] = VPUIP.SubView [[WEIGHTS_COPY]] [0, 0, 0, 0] [32, 64, 1, 1] : !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> to !VPUIP.DistributedBuffer<32x64x1x1xf16, {order = #NHWC, strides = [64, 1, 64, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       [[CAST0:%.*]] = VPUIP.DistributedCast inputs([[SUBVIEW0]] : !VPUIP.DistributedBuffer<32x64x1x1xf16, {order = #NHWC, strides = [64, 1, 64, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       [[SUBVIEW1:%.*]] = VPUIP.SubView %1 [32, 0, 0, 0] [32, 64, 1, 1] : !VPUIP.DistributedBuffer<64x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> to !VPUIP.DistributedBuffer<32x64x1x1xf16, {order = #NHWC, strides = [64, 1, 64, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       [[CAST1:%.*]] = VPUIP.DistributedCast inputs([[SUBVIEW1]] : !VPUIP.DistributedBuffer<32x64x1x1xf16, {order = #NHWC, strides = [64, 1, 64, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:       [[OUTBUF0:%.*]] = memref.alloc() : memref<32x64x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs([[CAST0]] as %arg1: memref<32x64x1x1xf16, #NHWC, @CMX_NN>) outputs([[OUTBUF0]] as %arg2: memref<32x64x1x1xf16, #NHWC, @DDR>) -> memref<32x64x1x1xf16, #NHWC, @DDR> {
    // CHECK:                                  VPUIP.Copy inputs(%arg1 : memref<32x64x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<32x64x1x1xf16, #NHWC, @DDR>) -> memref<32x64x1x1xf16, #NHWC, @DDR>
    // CHECK:       }
    // CHECK:       [[OUTBUF1:%.*]] = memref.alloc() : memref<32x64x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[CAST1]] as %arg1: memref<32x64x1x1xf16, #NHWC, @CMX_NN>) outputs([[OUTBUF1]] as %arg2: memref<32x64x1x1xf16, #NHWC, @DDR>) -> memref<32x64x1x1xf16, #NHWC, @DDR> {
    // CHECK:                                  VPUIP.Copy inputs(%arg1 : memref<32x64x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<32x64x1x1xf16, #NHWC, @DDR>) -> memref<32x64x1x1xf16, #NHWC, @DDR>
    // CHECK:       }
    // CHECK:       return [[COPY0]], [[COPY1]] : memref<32x64x1x1xf16, #NHWC, @DDR>, memref<32x64x1x1xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!WeightsType = memref<64x64x1x1xf16, #NHWC, @DDR>
!Output_DDR = memref<32x64x1x1xf16, #NHWC, @DDR>
!OutputStub_CMX = memref<32x64x1x1xf16, #NHWC, @CMX_NN>

func.func @NotMoveTilingCopyBeforeSubviewForSingleUser(%arg0: !WeightsType) -> !Output_DDR {
    %weights0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [32, 64, 1, 1] : memref<64x64x1x1xf16, #NHWC, @DDR> to memref<32x64x1x1xf16, #NHWC, @DDR>

    %weights0_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %weights0_copy = VPUIP.NCEClusterTiling inputs(%weights0 as %arg1: memref<32x64x1x1xf16, #NHWC>) outputs(%weights0_cmx as %arg2: memref<32x64x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<32x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
       %0 = VPUIP.Copy inputs(%arg1 : memref<32x64x1x1xf16, #NHWC>) outputs(%arg2 : memref<32x64x1x1xf16, #NHWC, @CMX_NN>) -> memref<32x64x1x1xf16, #NHWC, @CMX_NN>
    }

    // simulate nce task 0
    %output0_buf  = memref.alloc() : !Output_DDR
    %output0 = VPUIP.NCEClusterTiling inputs(%weights0_copy as %arg1: !OutputStub_CMX) outputs(%output0_buf as %arg2: !Output_DDR) -> !Output_DDR {
             VPUIP.Copy inputs(%arg1: !OutputStub_CMX) outputs(%arg2: !Output_DDR) -> !Output_DDR
    }

    return %output0: !Output_DDR

    // CHECK:       [[SUBVIEW:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [32, 64, 1, 1] : memref<64x64x1x1xf16, #NHWC, @DDR> to memref<32x64x1x1xf16, #NHWC, @DDR>
    // CHECK-NOT:   [[CAST:%.*]]  VPUIP.DistributedCast
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x256x1500x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>
!WeightsType = memref<1024x256x1x1xf16, #NHWC, @DDR>
!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x512x1500x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

func.func @NotMoveTilingCopyBeforeSubviewForNonSuitableCMXRequirements(%arg0: !WeightsType) -> (!OutputDistributedType, !OutputDistributedType) {
    %weights0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [512, 256, 1, 1] : !WeightsType to memref<512x256x1x1xf16, #NHWC, @DDR>
    %weights0_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<512x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %weights0_copy = VPUIP.NCEClusterTiling inputs(%weights0 as %arg1: memref<512x256x1x1xf16, #NHWC>) outputs(%weights0_cmx as %arg2: memref<512x256x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<512x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
            %0 = VPUIP.Copy inputs(%arg1 : memref<512x256x1x1xf16, #NHWC>) outputs(%arg2 : memref<512x256x1x1xf16, #NHWC, @CMX_NN>) -> memref<512x256x1x1xf16, #NHWC, @CMX_NN>
    }

    %input0 = VPURT.AllocDistributed -> !InputDistributedType
    %weights_table0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<512x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %output0_cmx = VPURT.AllocDistributed -> !OutputDistributedType
    // user nce task 0
    %nce_0 = VPUIP.NCEClusterTiling inputs(%input0 as %arg1: memref<1x256x1500x1xf16, #NHWC, @CMX_NN>, %weights0_copy as %arg2: memref<512x256x1x1xf16, #NHWC, @CMX_NN>, %weights_table0 as %arg3: memref<512x1x1x4xsi32, @CMX_NN>) outputs(%output0_cmx as %arg4: memref<1x512x1500x1xf16, #NHWC, @CMX_NN>) -> !OutputDistributedType {
            %0 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 102752 : i64, task_type = "CONV"} input(%arg1 : memref<1x256x1500x1xf16, #NHWC, @CMX_NN>) weights(%arg2 : memref<512x256x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg3 : memref<512x1x1x4xsi32, @CMX_NN>) parent_input(%arg1 : memref<1x256x1500x1xf16, #NHWC, @CMX_NN>) parent_output(%arg4 : memref<1x512x1500x1xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x512x1500x1xf16, #NHWC, @CMX_NN>) -> memref<1x512x1500x1xf16, #NHWC, @CMX_NN> variants : {
              DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [0, 1499, 255], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
              DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [0, 1499, 511], outStart = [0, 0, 256], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            } PPE : {
              PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
    }

    %weights1 = VPUIP.SubView %arg0 [512, 0, 0, 0] [512, 256, 1, 1] : !WeightsType to memref<512x256x1x1xf16, #NHWC, @DDR>
    %weights1_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<512x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %weights1_copy = VPUIP.NCEClusterTiling inputs(%weights1 as %arg1: memref<512x256x1x1xf16, #NHWC>) outputs(%weights0_cmx as %arg2: memref<512x256x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<512x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> {
            %0 = VPUIP.Copy inputs(%arg1 : memref<512x256x1x1xf16, #NHWC>) outputs(%arg2 : memref<512x256x1x1xf16, #NHWC, @CMX_NN>) -> memref<512x256x1x1xf16, #NHWC, @CMX_NN>
    }
    %input1 = VPURT.AllocDistributed -> !InputDistributedType
    %weights_table1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<512x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %output1_cmx = VPURT.AllocDistributed -> !OutputDistributedType
    // user nce task 0
    %nce_1 = VPUIP.NCEClusterTiling inputs(%input1 as %arg1: memref<1x256x1500x1xf16, #NHWC, @CMX_NN>, %weights1_copy as %arg2: memref<512x256x1x1xf16, #NHWC, @CMX_NN>, %weights_table1 as %arg3: memref<512x1x1x4xsi32, @CMX_NN>) outputs(%output1_cmx as %arg4: memref<1x512x1500x1xf16, #NHWC, @CMX_NN>) -> !OutputDistributedType {
            %0 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 102752 : i64, task_type = "CONV"} input(%arg1 : memref<1x256x1500x1xf16, #NHWC, @CMX_NN>) weights(%arg2 : memref<512x256x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg3 : memref<512x1x1x4xsi32, @CMX_NN>) parent_input(%arg1 : memref<1x256x1500x1xf16, #NHWC, @CMX_NN>) parent_output(%arg4 : memref<1x512x1500x1xf16, #NHWC, @CMX_NN>) outputs(%arg4 : memref<1x512x1500x1xf16, #NHWC, @CMX_NN>) -> memref<1x512x1500x1xf16, #NHWC, @CMX_NN> variants : {
              DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [0, 1499, 255], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
              DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [0, 1499, 511], outStart = [0, 0, 256], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
            } PPE : {
              PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
            }
    }

    return %nce_0, %nce_1: !OutputDistributedType, !OutputDistributedType

    // CHECK:       [[SUBVIEW0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [512, 256, 1, 1] : memref<1024x256x1x1xf16, #NHWC, @DDR> to memref<512x256x1x1xf16, #NHWC, @DDR>
    // CHECK-NOT:   VPUIP.DistributedCast
    // CHECK:       [[NCE_0:%.*]] = VPUIP.NCEClusterTask
    // CHECK:       [[SUBVIEW1:%.*]] = VPUIP.SubView %arg0 [512, 0, 0, 0] [512, 256, 1, 1] : memref<1024x256x1x1xf16, #NHWC, @DDR> to memref<512x256x1x1xf16, #NHWC, @DDR>
    // CHECK-NOT:   VPUIP.DistributedCast
    // CHECK:       [[NCE_1:%.*]] = VPUIP.NCEClusterTask
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x128x128xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 32, 35, 120], [1, 32, 35, 120]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 33, 0]]
}>

func.func @NoOptimizeDDRToCMXCopyWithOverlappedModeClusterCopy(%arg0: memref<1x3x128x128xf16, #NHWC, @DDR>)
                                    -> !OutputDistributed {
    %cst = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]
    %0 = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>

    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 3, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    %2 = VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>)
                outputs(%1 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 3, 0, 0] [1, 13, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR>
        to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    %4 = VPUIP.Copy inputs(%cst : memref<1x13x128x128xf16, #NHWC>)
                outputs(%3 : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>)
                        outputs(%0 : memref<1x16x128x128xf16, #NHWC, @DDR>) -> memref<1x16x128x128xf16, #NHWC, @DDR>

    %6 = VPURT.AllocDistributed -> !OutputDistributed
    %7 = VPUIP.NCEClusterTiling inputs(%5 as %arg2: memref<1x16x128x128xf16, #NHWC>) outputs(%6 as %arg3: memref<1x16x128x128xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
    %8 = VPUIP.Copy inputs(%arg2 : memref<1x16x128x128xf16, #NHWC>) outputs(%arg3 : memref<1x16x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x16x128x128xf16, #NHWC, @CMX_NN>
    }

    return %7 : !OutputDistributed

    // CHECK:   [[CST:%.*]] = const.Declare memref<1x13x128x128xf16, #NHWC> = dense<0.000000e+00> : tensor<212992xf16>, [#const.Reshape<[1, 13, 128, 128]>, #const.Reorder<#NHWC>]
    // CHECK:   [[CONCAT_BUFF:%.*]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, @DDR>
    // CHECK:   [[SUBVIEW_0:%.*]] = VPUIP.SubView [[CONCAT_BUFF]] [0, 0, 0, 0] [1, 3, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR> to memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    // CHECK:   [[COPY_0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x3x128x128xf16, #NHWC, @DDR>) outputs([[SUBVIEW_0]] : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    // CHECK:   [[SUBVIEW_1:%.*]] = VPUIP.SubView [[CONCAT_BUFF]] [0, 3, 0, 0] [1, 13, 128, 128] : memref<1x16x128x128xf16, #NHWC, @DDR> to memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>
    // CHECK:   [[COPY_1:%.*]] = VPUIP.Copy inputs([[CST]] : memref<1x13x128x128xf16, #NHWC>) outputs([[SUBVIEW_1]] : memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) -> memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>

    // CHECK:   [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : memref<1x3x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>, memref<1x13x128x128xf16, {order = #NHWC, strides = [262144, 1, 2048, 16]}, @DDR>) outputs([[CONCAT_BUFF]] : memref<1x16x128x128xf16, #NHWC, @DDR>) -> memref<1x16x128x128xf16, #NHWC, @DDR>
    // CHECK:   [[OUT_BUFF:%.*]] = VPURT.AllocDistributed ->
    // CHEKC-SAME         !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 32, 35, 120], [1, 32, 35, 120]], compute_offsets = [[0, 0, 0, 0], [0, 0, 33, 0]]}>

    // CHECK:   [[COPY_2:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x16x128x128xf16, #NHWC>) outputs([[OUT_BUFF]] as %arg2: memref<1x16x128x128xf16, #NHWC, @CMX_NN>) ->
    // CHEKC-SAME           !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 32, 35, 120], [1, 32, 35, 120]], compute_offsets = [[0, 0, 0, 0], [0, 0, 33, 0]]}>
    // CHECK:   return [[COPY_2]] :
    // CHEKC-SAME   !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 32, 35, 120], [1, 32, 35, 120]], compute_offsets = [[0, 0, 0, 0], [0, 0, 33, 0]]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!qElemType = !quant.uniform<u8:f16, 1.0:123>
!qElemType1 = !quant.uniform<u8:f16, 2.0:123>


func.func @DDR2DDRCopyMultiInputsWithDifferentType(%in : memref<1x144x128x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>,
                       %arg1: memref<1x144x64x128x!qElemType1, #NHWC>,
                       %weights: memref<32x144x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>,
                       %weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
                        -> (!OutputDistributed,  memref<1x144x64x128x!qElemType1, #NHWC, @DDR>) {
    %0 = VPUIP.SubView %in [0, 0, 0, 0] [1, 144, 64, 128]
            : memref<1x144x128x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
            to memref<1x144x64x128x!qElemType, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %1 = memref.alloc() : memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %2 = VPUIP.Copy
            inputs(%0 : memref<1x144x64x128x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>)
            outputs(%1 : memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
                -> memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
    %4 = VPUIP.QuantizeCast
        inputs(%2 : memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>)
        -> memref<1x144x64x128x!qElemType1, #NHWC, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %7 = VPUIP.NCEClusterTiling
            inputs(%2 as %arg2: memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>)
            outputs(%3 as %arg3: memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                -> !OutputDistributed {
        %inner = VPUIP.Copy
                inputs(%arg2 : memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>)
                outputs(%arg3 : memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                    -> memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>
    }
    %5 = VPURT.AllocDistributed -> !OutputDistributed
    %6 = VPUIP.NCEClusterTiling
            inputs(
                %7 as %arg2: memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>,
                %weights as %arg3: memref<32x144x1x1x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>,
                %weights_table as %arg4: memref<32x1x1x4xsi32, @CMX_NN>)
            outputs(
                %5 as %arg5: memref<1x32x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                    -> !OutputDistributed {
        %inner = VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = "CONV"}
            input(
                %arg2 : memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                weights(%arg3 : memref<32x144x1x1x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                weight_table(%arg4 : memref<32x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x144x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                parent_output(%arg5 : memref<1x32x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                outputs(%arg5 : memref<1x32x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
                    -> memref<1x32x64x128x!qElemType, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
        } PPE :  {
        }
    }

    return %6 , %4 : !OutputDistributed, memref<1x144x64x128x!qElemType1, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128x!qElemType0, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128x!qElemType0, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    // CHECK:       [[COPY_BUFF:%.*]] = memref.alloc() : memref<1x144x64x128x!qElemType0, #NHWC, @DDR>
    // CHECK:       [[DDRCOPY:%.*]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x144x64x128x!qElemType0, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
    // CHECK-SAME:     outputs([[COPY_BUFF]] : memref<1x144x64x128x!qElemType0, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x144x64x128x!qElemType0, #NHWC, @DDR>

    // CHECK:       [[QC:%.*]] = VPUIP.QuantizeCast inputs(
    // CHECK-SAME:      [[DDRCOPY]] : memref<1x144x64x128x!qElemType0, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x144x64x128x!qElemType1, #NHWC, @DDR>

    // CHECK:       [[BUFFER_1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[DDRCOPY]] as %arg4: memref<1x144x64x128x!qElemType0, #NHWC>)
    // CHECK-SAME:      outputs([[BUFFER_1]] as %arg5: memref<1x144x64x128x!qElemType0, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    // CHECK:       [[COPY_INNER:%.*]] = VPUIP.Copy inputs(%arg4 : memref<1x144x64x128x!qElemType0, #NHWC>)
    // CHECK-SAME:      outputs(%arg5 : memref<1x144x64x128x!qElemType0, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> memref<1x144x64x128x!qElemType0, #NHWC, @CMX_NN>

    // CHECK:       [[BUFFER_2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[NCE:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[COPY]] as %arg4: memref<1x144x64x128x!qElemType0, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg2 as %arg5: memref<32x144x1x1x!qElemType0, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg3 as %arg6: memref<32x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_2]] as %arg7: memref<1x32x64x128x!qElemType0, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       return [[NCE]], [[QC]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, memref<1x144x64x128x!qElemType1, #NHWC, @DDR>
}

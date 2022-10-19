// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-copies %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @OptimizeCopy(
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

func @OptimizeLastCopyForPureViewOps(%arg0: memref<1x16x2x2xf16>, %arg1: memref<1x16x2x2xf16>, %arg2: memref<1x64x2x2xf16>) -> memref<1x64x2x2xf16> {
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

func @NoChangesSourceIsConstantOp(%arg0: memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16> {
    %0 = const.Declare memref<1x2x4x4xf16> = #const.Content<dense<1.000000e+00> : tensor<1x2x4x4xf16>>
    %1 = VPUIP.Copy inputs(%0 : memref<1x2x4x4xf16>) outputs(%arg0 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>
    return %1 : memref<1x2x4x4xf16>

    // CHECK: [[VAR0:%.*]] = const.Declare
    // CHECK: [[VAR1:%.*]] = VPUIP.Copy
    // CHECK: return [[VAR1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @NoChangesDifferentMemSpace(%arg0: memref<1x16x4x4xf16, #NHWC>, %arg1 : memref<16x1x1x4xsi32, @CMX_NN>,
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
            DPUTask { end = [16, 2, 2], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
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

func @CopiesWithSubViewOps(%act : memref<1x80x28x28xf16, #NHWC, @DDR>)
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
    // CHECK:       [[VAR3:%.*]] = VPUIP.Copy inputs(%2 : memref<1x70x28x27xf16, {order = #NHWC, strides = [62720, 1, 2240, 80]}, @DDR>) outputs(%0 : memref<1x70x28x27xf16, #NHWC, @DDR>) ->
    // CHECK-SAME:      memref<1x70x28x27xf16, #NHWC, @DDR>
    // CHECK:       [[VAR4:%.*]] = VPUIP.SubView [[VAR1]] [0, 0, 0, 0] [1, 70, 28, 27] :
    // CHECK-SAME:      memref<1x80x28x27xf16, #NHWC, @DDR> to memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    // CHECK:       [[VAR5:%.*]] = VPUIP.Copy inputs(%3 : memref<1x70x28x27xf16, #NHWC, @DDR>) outputs(%4 : memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>) ->
    // CHECK-SAME:      memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    // CHECK:       [[VAR6:%.*]] = VPUIP.SubView [[VAR3]] [0, 0, 0, 0] [1, 10, 28, 27] :
    // CHECK-SAME:      memref<1x70x28x27xf16, #NHWC, @DDR> to memref<1x10x28x27xf16, {order = #NHWC, strides = [52920, 1, 1890, 70]}, @DDR>
    // CHECK:       [[VAR7:%.*]] = VPUIP.SubView [[VAR1]] [0, 70, 0, 0] [1, 10, 28, 27] :
    // CHECK-SAME:      memref<1x80x28x27xf16, #NHWC, @DDR> to memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    // CHECK:       [[VAR8:%.*]] = VPUIP.Copy inputs(%6 : memref<1x10x28x27xf16, {order = #NHWC, strides = [52920, 1, 1890, 70]}, @DDR>) outputs(%7 : memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>) ->
    // CHECK-SAME:      memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>
    // CHECK:       [[VAR9:%.*]] = VPUIP.ConcatView inputs(%5, %8 : memref<1x70x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>, memref<1x10x28x27xf16, {order = #NHWC, strides = [60480, 1, 2160, 80]}, @DDR>)
    // CHECK-SAME:      outputs(%1 : memref<1x80x28x27xf16, #NHWC, @DDR>) -> memref<1x80x28x27xf16, #NHWC, @DDR>
    // CHECK:       return %9 : memref<1x80x28x27xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = type !quant.uniform<u8:f16, 1.000000e+00>

func @CopyWithQuantizeCastOp(%arg0: memref<1x16x56x56x!qElemType, #NHWC, @CMX_NN>,
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

func @CopyWithSubViewOp(%in : memref<1x16x113x113xf16, #NHWC, @DDR>,
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
                    end = [55, 10, 15], mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    start = [0, 0, 0]
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
    // CHECK-NOT:   IE.Copy

    // CHECK:       return [[VAL3:%.+]] : memref<1x16x56x56xf16, {order = #NHWC, strides = [100352, 1, 1792, 32]}, @CMX_NN>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @DDR2DDRCopyOutput(%in : memref<1x32x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN>)
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

    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView %0 [0, 0, 0, 0] [1, 32, 64, 128] :
    // CHECK-SAME:      memref<1x32x128x128xf16, #NHWC, @DDR> to memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR>
    // CHECK:       [[COPY_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs(%arg0 as %arg1: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[SUBVIEW_1]] as %arg2: memref<1x32x64x128xf16, #NHWC>)
    // CHECK-SAME:          -> memref<1x32x64x128xf16, {order = #NHWC, strides = [524288, 1, 4096, 32]}, @DDR> {
    // CHECK:       [[COPY_1_INNER:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs(%arg2 : memref<1x32x64x128xf16, #NHWC>)
    // CHECK-SAME:          -> memref<1x32x64x128xf16, #NHWC>

    // CHECK:       [[SUBVIEW_2:%.*]] = VPUIP.SubView %0 [0, 0, 64, 0] [1, 32, 64, 128] :
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
    // CHECK-SAME:      outputs(%0 : memref<1x32x128x128xf16, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x32x128x128xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCAT]] : memref<1x32x128x128xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x144x64x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

func @DDR2DDRCopyInput(%in : memref<1x144x128x128xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>,
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
                DPUTask {cluster_id = 0 : i64, end = [15, 5, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
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
!qElemType0 = type !quant.uniform<u8:f16, 5.7832517137714463:123>
!qElemType1 = type !quant.uniform<u8:f16, 7.1335556927849266:124>

func @ParallelDDR2DDRCopyOutput(%in0 : memref<1x64x48x88x!qElemType0, #NHWC, @CMX_NN>,
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
                DPUTask {cluster_id = 0 : i64, end = [87, 47, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
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
    // CHECK:                   DPUTask {cluster_id = 0 : i64, end = [87, 47, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
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
!qElemType = type !quant.uniform<u8:f16, 5.7832517137714463:123>

func @DDR2DDRCopyOutputNOSubview(%in0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
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
                DPUTask {cluster_id = 0 : i64, end = [87, 47, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
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
    // CHECK:                   DPUTask {cluster_id = 0 : i64, end = [87, 47, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
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

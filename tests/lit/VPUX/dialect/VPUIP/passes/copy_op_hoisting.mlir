// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --copy-op-hoisting %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CopyFromBlockArgumentNoChange
func @CopyFromBlockArgumentNoChange(%arg0: memref<1x16x1x1xf16, @CMX_NN>, %arg1: memref<1x16x1x1xf16>, %arg2: memref<1x16x1x1xf16>)
        -> (memref<1x16x1x1xf16>, memref<1x16x1x1xf16>) {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x16x1x1xf16, @CMX_NN>) outputs(%arg1 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x1x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    return %0, %1 : memref<1x16x1x1xf16>, memref<1x16x1x1xf16>

    // CHECK: [[VAR0:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x16x1x1xf16, @CMX_NN>) outputs(%arg1 : memref<1x16x1x1xf16>)
    // CHECK: [[VAR1:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x16x1x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x1x1xf16>)
    // CHECK: return [[VAR0]], [[VAR1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CopyToBlockArgumentNoChange
func @CopyToBlockArgumentNoChange(%arg0: memref<1x16x1x1xf16, #NHWC, @CMX_NN>, %arg1: memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16> {
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = #const.Content<dense<1> : tensor<1x1x1x16xui8>>
    %0 = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        parent_output(%0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x16x1x1xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { end = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
        }
        PPE : {
        }

    %2 = VPUIP.Copy inputs(%1 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    return %2 : memref<1x16x1x1xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR0]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR2:%.+]] = VPUIP.Copy inputs([[VAR1]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x16x1x1xf16>)
    // CHECK: return [[VAR2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CopyToBlockArgumentMoveCopyOnly
func @CopyToBlockArgumentMoveCopyOnly(%arg0: memref<1x16x1x1xf16, #NHWC, @CMX_NN>, %arg1: memref<1x16x1x1xf16>, %arg2: memref<1x16x1x1xf16>) -> (memref<1x16x1x1xf16>, memref<1x16x1x1xf16>) {
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = #const.Content<dense<1> : tensor<1x1x1x16xui8>>
    %0 = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    %1 = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    %2 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        parent_output(%0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x16x1x1xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { end = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
        }
        PPE : {
        }
    %3 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        parent_output(%1 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        outputs(%1 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x16x1x1xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { end = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
        }
        PPE : {
        }
    %4 = VPUIP.Copy inputs(%2 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    %5 = VPUIP.Copy inputs(%3 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    return %4, %5 : memref<1x16x1x1xf16>, memref<1x16x1x1xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR0]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x16x1x1xf16>)
    // CHECK: [[VAR4:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR1]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR5:%.+]] = VPUIP.Copy inputs([[VAR4]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x1x1xf16>)
    // CHECK: return [[VAR3]], [[VAR5]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CopyToBlockArgumentSubView
func @CopyToBlockArgumentSubView(%arg0: memref<1x8x1x1xf16, #NHWC, @CMX_NN>, %arg1: memref<16x1x1xf16>) -> memref<16x1x1xf16> {
    %wt = const.Declare memref<8x1x1x4xsi32, @CMX_NN> = #const.Content<dense<1> : tensor<8x1x1x4xsi32>>
    %act_win = const.Declare memref<1x1x1x8xui8, @CMX_NN> = #const.Content<dense<1> : tensor<1x1x1x8xui8>>
    %0 = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    %1 = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    %2 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<8x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x8xui8, @CMX_NN>)
        parent_input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
        parent_output(%0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x8x1x1xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { end = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
        }
        PPE : {
        }
    %3 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = "MAXPOOL"
        }
        input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<8x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x8xui8, @CMX_NN>)
        parent_input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
        parent_output(%1 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
        outputs(%1 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x8x1x1xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { end = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
        }
        PPE : {
        }
    %4 = VPUIP.SubView %arg1 [0, 0, 0] [8, 1, 1] : memref<16x1x1xf16> to memref<8x1x1xf16>
    %5 = VPUIP.Copy inputs(%2 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs(%4 : memref<8x1x1xf16>) -> memref<8x1x1xf16>
    %6 = VPUIP.SubView %arg1 [8, 0, 0] [8, 1, 1] : memref<16x1x1xf16> to memref<8x1x1xf16>
    %7 = VPUIP.Copy inputs(%3 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs(%6 : memref<8x1x1xf16>) -> memref<8x1x1xf16>
    %8 = VPUIP.ConcatView inputs(%5, %7 : memref<8x1x1xf16>, memref<8x1x1xf16>) outputs(%arg1 : memref<16x1x1xf16>) -> memref<16x1x1xf16>
    return %8 : memref<16x1x1xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR0]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR3:%.+]] = VPUIP.SubView %arg1 [0, 0, 0] [8, 1, 1]
    // CHECK: [[VAR4:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs([[VAR3]] : memref<8x1x1xf16>)
    // CHECK: [[VAR5:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR1]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR6:%.+]] = VPUIP.SubView %arg1 [8, 0, 0] [8, 1, 1]
    // CHECK: [[VAR7:%.+]] = VPUIP.Copy inputs([[VAR5]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs([[VAR6]] : memref<8x1x1xf16>)
    // CHECK: [[VAR8:%.+]] = VPUIP.ConcatView inputs([[VAR4]], [[VAR7]] : memref<8x1x1xf16>, memref<8x1x1xf16>) outputs(%arg1 : memref<16x1x1xf16>)
    // CHECK: return [[VAR8]]
}

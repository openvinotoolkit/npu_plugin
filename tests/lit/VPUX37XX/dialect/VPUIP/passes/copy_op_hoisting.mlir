// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --copy-op-hoisting %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @CopyToTempBufNoChange
func @CopyToTempBufNoChange(%arg0: memref<1x16x1x1xf16, #NHWC, @CMX_NN>, %arg1: memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16> {
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
    %2 = memref.alloc() : memref<1x16x1x1xf16>
    %3 = VPUIP.Copy inputs(%1 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%2 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    %4 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
        @VPU.SW::@builtin_Sigmoid inputs(%3: memref<1x16x1x1xf16>) outputs(%arg1 : memref<1x16x1x1xf16>) on tile 0 -> memref<1x16x1x1xf16>  {
        ^bb0(%arg3: memref<1x16x1x1xf16>, %arg4: memref<1x16x1x1xf16>):
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x16x1x1xf16>, memref<1x16x1x1xf16>
        }
    return %4 : memref<1x16x1x1xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR0]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR2:%.+]] = memref.alloc() : memref<1x16x1x1xf16>
    // CHECK: [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR1]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs([[VAR2]] : memref<1x16x1x1xf16>)
    // CHECK: [[VAR4:%.+]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Sigmoid inputs([[VAR3]] : memref<1x16x1x1xf16>) outputs(%arg1 : memref<1x16x1x1xf16>)
    // CHECK: return [[VAR4]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @CopyToTempBufMoveCopyOnly
func @CopyToTempBufMoveCopyOnly(%arg0: memref<1x16x1x1xf16, #NHWC, @CMX_NN>, %arg1: memref<1x16x1x1xf16>, %arg2: memref<1x16x1x1xf16>) -> (memref<1x16x1x1xf16>, memref<1x16x1x1xf16>) {
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = #const.Content<dense<1> : tensor<1x1x1x16xui8>>
    %0 = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    %1 = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    %2 = memref.alloc() : memref<1x16x1x1xf16>
    %3 = memref.alloc() : memref<1x16x1x1xf16>
    %4 = VPUIP.NCEClusterTask {
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
    %5 = VPUIP.NCEClusterTask {
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
    %6 = VPUIP.Copy inputs(%4 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%2 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    %7 = VPUIP.Copy inputs(%5 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%3 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    %8 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
        @VPU.SW::@builtin_Sigmoid inputs(%6: memref<1x16x1x1xf16>) outputs(%arg1 : memref<1x16x1x1xf16>) on tile 0 -> memref<1x16x1x1xf16>  {
        ^bb0(%arg3: memref<1x16x1x1xf16>, %arg4: memref<1x16x1x1xf16>):
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x16x1x1xf16>, memref<1x16x1x1xf16>
        }
    %9 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
        @VPU.SW::@builtin_Sigmoid inputs(%7: memref<1x16x1x1xf16>) outputs(%arg2 : memref<1x16x1x1xf16>) on tile 0 -> memref<1x16x1x1xf16>  {
        ^bb0(%arg3: memref<1x16x1x1xf16>, %arg4: memref<1x16x1x1xf16>):
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x16x1x1xf16>, memref<1x16x1x1xf16>
        }
    return %8, %9 : memref<1x16x1x1xf16>, memref<1x16x1x1xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR2:%.+]] = memref.alloc() : memref<1x16x1x1xf16>
    // CHECK: [[VAR3:%.+]] = memref.alloc() : memref<1x16x1x1xf16>
    // CHECK: [[VAR4:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR0]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR5:%.+]] = VPUIP.Copy inputs([[VAR4]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs([[VAR2]] : memref<1x16x1x1xf16>)
    // CHECK: [[VAR6:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR1]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR7:%.+]] = VPUIP.Copy inputs([[VAR6]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs([[VAR3]] : memref<1x16x1x1xf16>)
    // CHECK: [[VAR8:%.+]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Sigmoid inputs([[VAR5]] : memref<1x16x1x1xf16>) outputs(%arg1 : memref<1x16x1x1xf16>)
    // CHECK: [[VAR9:%.+]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Sigmoid inputs([[VAR7]] : memref<1x16x1x1xf16>) outputs(%arg2 : memref<1x16x1x1xf16>)
    // CHECK: return [[VAR8]], [[VAR9]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @CopyToTempBufMoveWithAlloc
func @CopyToTempBufMoveWithAlloc(%arg0: memref<1x16x1x1xf16, #NHWC, @CMX_NN>, %arg1: memref<1x16x1x1xf16>, %arg2: memref<1x16x1x1xf16>) -> (memref<1x16x1x1xf16>, memref<1x16x1x1xf16>) {
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
    %2 = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
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
        parent_output(%2 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
        outputs(%2 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x16x1x1xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { end = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
        }
        PPE : {
        }
    %4 = memref.alloc() : memref<1x16x1x1xf16>
    %5 = VPUIP.Copy inputs(%1 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%4 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    %6 = memref.alloc() : memref<1x16x1x1xf16>
    %7 = VPUIP.Copy inputs(%3 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs(%6 : memref<1x16x1x1xf16>) -> memref<1x16x1x1xf16>
    %8 = VPUIP.SW.Kernel{result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
        @VPU.SW::@builtin_Sigmoid inputs(%5: memref<1x16x1x1xf16>) outputs(%arg1 : memref<1x16x1x1xf16>) on tile 0 -> memref<1x16x1x1xf16>  {
        ^bb0(%arg3: memref<1x16x1x1xf16>, %arg4: memref<1x16x1x1xf16>):
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x16x1x1xf16>, memref<1x16x1x1xf16>
        }
    %9 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
        @VPU.SW::@builtin_Sigmoid inputs(%7: memref<1x16x1x1xf16>) outputs(%arg2 : memref<1x16x1x1xf16>) on tile 0 -> memref<1x16x1x1xf16>  {
        ^bb0(%arg3: memref<1x16x1x1xf16>, %arg4: memref<1x16x1x1xf16>):
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x16x1x1xf16>, memref<1x16x1x1xf16>
        }
    return %8, %9 : memref<1x16x1x1xf16>, memref<1x16x1x1xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR0]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR2:%.+]] = memref.alloc() : memref<1x16x1x1xf16>
    // CHECK: [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR1]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs([[VAR2]] : memref<1x16x1x1xf16>)
    // CHECK: [[VAR4:%.+]] = memref.alloc() : memref<1x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR5:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR4]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR6:%.+]] = memref.alloc() : memref<1x16x1x1xf16>
    // CHECK: [[VAR7:%.+]] = VPUIP.Copy inputs([[VAR5]] : memref<1x16x1x1xf16, #NHWC, @CMX_NN>) outputs([[VAR6]] : memref<1x16x1x1xf16>)
    // CHECK: [[VAR8:%.+]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Sigmoid inputs([[VAR3]] : memref<1x16x1x1xf16>) outputs(%arg1 : memref<1x16x1x1xf16>)
    // CHECK: [[VAR9:%.+]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Sigmoid inputs([[VAR7]] : memref<1x16x1x1xf16>) outputs(%arg2 : memref<1x16x1x1xf16>)
    // CHECK: return [[VAR8]], [[VAR9]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @CopyToTempBufSubViewMoveCopyOnly
func @CopyToTempBufSubViewMoveCopyOnly(%arg0: memref<1x8x1x1xf16, #NHWC, @CMX_NN>, %arg1: memref<16x1x1xf16>) -> memref<16x1x1xf16> {
    %wt = const.Declare memref<8x1x1x4xsi32, @CMX_NN> = #const.Content<dense<1> : tensor<8x1x1x4xsi32>>
    %act_win = const.Declare memref<1x1x1x8xui8, @CMX_NN> = #const.Content<dense<1> : tensor<1x1x1x8xui8>>
    %0 = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    %1 = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    %2 = memref.alloc() : memref<16x1x1xf16>
    %3 = VPUIP.SubView %2 [0, 0, 0] [8, 1, 1] : memref<16x1x1xf16> to memref<8x1x1xf16>
    %4 = VPUIP.SubView %2 [8, 0, 0] [8, 1, 1] : memref<16x1x1xf16> to memref<8x1x1xf16>
    %5 = VPUIP.NCEClusterTask {
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
    %6 = VPUIP.NCEClusterTask {
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
    %7 = VPUIP.Copy inputs(%5 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs(%3 : memref<8x1x1xf16>) -> memref<8x1x1xf16>
    %8 = VPUIP.Copy inputs(%6 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs(%4 : memref<8x1x1xf16>) -> memref<8x1x1xf16>
    %9 = VPUIP.ConcatView inputs(%7, %8 : memref<8x1x1xf16>, memref<8x1x1xf16>) outputs(%2 : memref<16x1x1xf16>) -> memref<16x1x1xf16>
    %10 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
        @VPU.SW::@builtin_Sigmoid inputs(%9: memref<16x1x1xf16>) outputs(%arg1 : memref<16x1x1xf16>) on tile 0 -> memref<16x1x1xf16>  {
        ^bb0(%arg3: memref<16x1x1xf16>, %arg4: memref<16x1x1xf16>):
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<16x1x1xf16>, memref<16x1x1xf16>
        }
    return %10 : memref<16x1x1xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR2:%.+]] = memref.alloc() : memref<16x1x1xf16>
    // CHECK: [[VAR3:%.+]] = VPUIP.SubView [[VAR2]] [0, 0, 0] [8, 1, 1]
    // CHECK: [[VAR4:%.+]] = VPUIP.SubView [[VAR2]] [8, 0, 0] [8, 1, 1]
    // CHECK: [[VAR5:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR0]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR6:%.+]] = VPUIP.Copy inputs([[VAR5]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs([[VAR3]] : memref<8x1x1xf16>)
    // CHECK: [[VAR7:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR1]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR8:%.+]] = VPUIP.Copy inputs([[VAR7]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs([[VAR4]] : memref<8x1x1xf16>)
    // CHECK: [[VAR9:%.+]] = VPUIP.ConcatView inputs([[VAR6]], [[VAR8]] : memref<8x1x1xf16>, memref<8x1x1xf16>) outputs([[VAR2]] : memref<16x1x1xf16>)
    // CHECK: [[VAR10:%.+]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Sigmoid inputs([[VAR9]] : memref<16x1x1xf16>) outputs(%arg1 : memref<16x1x1xf16>)
    // CHECK: return [[VAR10]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @CopyToTempBufSubViewMoveWithAlloc
func @CopyToTempBufSubViewMoveWithAlloc(%arg0: memref<1x8x1x1xf16, #NHWC, @CMX_NN>, %arg1: memref<16x1x1xf16>) -> memref<16x1x1xf16> {
    %wt = const.Declare memref<8x1x1x4xsi32, @CMX_NN> = #const.Content<dense<1> : tensor<8x1x1x4xsi32>>
    %act_win = const.Declare memref<1x1x1x8xui8, @CMX_NN> = #const.Content<dense<1> : tensor<1x1x1x8xui8>>
    %0 = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTask {
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
    %2 = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
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
        parent_output(%2 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
        outputs(%2 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x8x1x1xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { end = [16, 1, 1], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0] }
        }
        PPE : {
        }
    %4 = memref.alloc() : memref<16x1x1xf16>
    %5 = VPUIP.SubView %4 [0, 0, 0] [8, 1, 1] : memref<16x1x1xf16> to memref<8x1x1xf16>
    %6 = VPUIP.Copy inputs(%1 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs(%5 : memref<8x1x1xf16>) -> memref<8x1x1xf16>
    %7 = VPUIP.SubView %4 [8, 0, 0] [8, 1, 1] : memref<16x1x1xf16> to memref<8x1x1xf16>
    %8 = VPUIP.Copy inputs(%3 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs(%7 : memref<8x1x1xf16>) -> memref<8x1x1xf16>
    %9 = VPUIP.ConcatView inputs(%6, %8 : memref<8x1x1xf16>, memref<8x1x1xf16>) outputs(%4 : memref<16x1x1xf16>) -> memref<16x1x1xf16>
    %10 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
        @VPU.SW::@builtin_Sigmoid inputs(%9: memref<16x1x1xf16>) outputs(%arg1 : memref<16x1x1xf16>) on tile 0 -> memref<16x1x1xf16>  {
        ^bb0(%arg3: memref<16x1x1xf16>, %arg4: memref<16x1x1xf16>):
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<16x1x1xf16>, memref<16x1x1xf16>
        }
    return %10 : memref<16x1x1xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR0]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR2:%.+]] = memref.alloc() : memref<16x1x1xf16>
    // CHECK: [[VAR3:%.+]] = VPUIP.SubView [[VAR2]] [0, 0, 0] [8, 1, 1]
    // CHECK: [[VAR4:%.+]] = VPUIP.Copy inputs([[VAR1]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs([[VAR3]] : memref<8x1x1xf16>)
    // CHECK: [[VAR5:%.+]] = memref.alloc() : memref<1x8x1x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[VAR6:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input(%arg0 : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[VAR5]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK: [[VAR7:%.+]] = VPUIP.SubView [[VAR2]] [8, 0, 0] [8, 1, 1]
    // CHECK: [[VAR8:%.+]] = VPUIP.Copy inputs([[VAR6]] : memref<1x8x1x1xf16, #NHWC, @CMX_NN>) outputs([[VAR7]] : memref<8x1x1xf16>)
    // CHECK: [[VAR9:%.+]] = VPUIP.ConcatView inputs([[VAR4]], [[VAR8]] : memref<8x1x1xf16>, memref<8x1x1xf16>) outputs([[VAR2]] : memref<16x1x1xf16>)
    // CHECK: [[VAR10:%.+]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Sigmoid inputs([[VAR9]] : memref<16x1x1xf16>) outputs(%arg1 : memref<16x1x1xf16>)
    // CHECK: return [[VAR10]]
}

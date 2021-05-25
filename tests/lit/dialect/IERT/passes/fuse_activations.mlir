// RUN: vpux-opt --split-input-file --fuse-activations %s | FileCheck %s

func @MLIRConv2dWithReluTest(%arg0: memref<1x16x4x4xf32>, %arg1: memref<1x16x3x3xf32>) -> memref<1x16x3x3xf32> {
    %0 = IERT.Constant memref<16x16x2x2xf16> = opaque<"_", "0xDEADBEEF"> : tensor<16x16x2x2xf32>
    %1 = memref.alloc() : memref<1x16x4x4xf16>
    %2 = IERT.Convert inputs(%arg0 : memref<1x16x4x4xf32>) outputs(%1 : memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16>
    %3 = memref.alloc() : memref<1x16x3x3xf16>
    %4 = memref.alloc() : memref<16x16x2x2xf16>
    %5 = IERT.Reorder inputs(%0 : memref<16x16x2x2xf16>) outputs(%4 : memref<16x16x2x2xf16>) -> memref<16x16x2x2xf16>
    %6 = memref.alloc() : memref<16x16x2x2xf16, "CMX_NN">
    %7 = IERT.Copy inputs(%5 : memref<16x16x2x2xf16>) outputs(%6 : memref<16x16x2x2xf16, "CMX_NN">) -> memref<16x16x2x2xf16, "CMX_NN">
    %8 = IERT.Constant memref<16x1x1x4xsi32> = opaque<"_", "0xDEADBEEF"> : tensor<16x1x1x4xsi32>
    %9 = memref.alloc() : memref<16x1x1x4xsi32, "CMX_NN">
    %10 = IERT.Copy inputs(%8 : memref<16x1x1x4xsi32>) outputs(%9 : memref<16x1x1x4xsi32, "CMX_NN">) -> memref<16x1x1x4xsi32, "CMX_NN">
    %11 = memref.alloc() : memref<1x16x4x4xf16>
    %12 = IERT.Reorder inputs(%2 : memref<1x16x4x4xf16>) outputs(%11 : memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16>
    %13 = memref.alloc() : memref<1x16x4x4xf16, "CMX_NN">
    %14 = IERT.Copy inputs(%12 : memref<1x16x4x4xf16>) outputs(%13 : memref<1x16x4x4xf16, "CMX_NN">) -> memref<1x16x4x4xf16, "CMX_NN">
    %15 = memref.alloc() : memref<1x16x3x3xf16, "CMX_NN">
    %16 = VPUIP.NCEClusterTask {fixed_ppe_task = "NOOP", kernel_padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], kernel_size = [2 : i32, 2 : i32], strides = [1 : i32, 1 : i32], task_type = "CONV"} inputs(%14 : memref<1x16x4x4xf16, "CMX_NN">, %7 : memref<16x16x2x2xf16, "CMX_NN">, %10 : memref<16x1x1x4xsi32, "CMX_NN">)) parent_input(%14 : memref<1x16x4x4xf16, "CMX_NN">) parent_output(%15 : memref<1x16x3x3xf16, "CMX_NN">) outputs(%15 : memref<1x16x3x3xf16, "CMX_NN">) variants :  {
      VPUIP.DPUTask {end = [2 : i32, 2 : i32, 15 : i32], mpe_mode = "VECTOR_FP16", pads_begin = [0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32], start = [0 : i32, 0 : i32, 0 : i32]}
    } -> memref<1x16x3x3xf16, "CMX_NN">
    %17 = memref.alloc() : memref<1x16x3x3xf16>
    %18 = IERT.Copy inputs(%16 : memref<1x16x3x3xf16, "CMX_NN">) outputs(%17 : memref<1x16x3x3xf16>) -> memref<1x16x3x3xf16>
    %19 = IERT.Reorder inputs(%18 : memref<1x16x3x3xf16>) outputs(%3 : memref<1x16x3x3xf16>) -> memref<1x16x3x3xf16>
    %20 = memref.alloc() : memref<1x16x3x3xf16>
    %21 = IERT.ReLU inputs(%19 : memref<1x16x3x3xf16>) outputs(%20 : memref<1x16x3x3xf16>) -> memref<1x16x3x3xf16>
    %22 = memref.alloc() : memref<1x16x3x3xf32>
    %23 = IERT.Convert inputs(%21 : memref<1x16x3x3xf16>) outputs(%22 : memref<1x16x3x3xf32>) -> memref<1x16x3x3xf32>
    %24 = IERT.Copy inputs(%23 : memref<1x16x3x3xf32>) outputs(%arg1 : memref<1x16x3x3xf32>) -> memref<1x16x3x3xf32>
    return %24 : memref<1x16x3x3xf32>
}

// CHECK:    %16 = VPUIP.NCEClusterTask {fixed_ppe_task = "LRELU", kernel_padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], kernel_size = [2 : i32, 2 : i32], strides = [1 : i32, 1 : i32], task_type = "CONV"} inputs(%14 : memref<1x16x4x4xf16, "CMX_NN">, %7 : memref<16x16x2x2xf16, "CMX_NN">, %10 : memref<16x1x1x4xsi32, "CMX_NN">)) parent_input(%14 : memref<1x16x4x4xf16, "CMX_NN">) parent_output(%15 : memref<1x16x3x3xf16, "CMX_NN">) outputs(%15 : memref<1x16x3x3xf16, "CMX_NN">) variants :  {
// CHECK:      VPUIP.DPUTask {end = [2 : i32, 2 : i32, 15 : i32], mpe_mode = "VECTOR_FP16", pads_begin = [0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32], start = [0 : i32, 0 : i32, 0 : i32]}
// CHECK:    } -> memref<1x16x3x3xf16, "CMX_NN">
// CHECK-NOT: %21 = IERT.ReLU {{.*}}


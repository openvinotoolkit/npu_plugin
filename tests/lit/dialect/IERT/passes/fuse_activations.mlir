// RUN: vpux-opt --split-input-file --fuse-activations %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @MLIRConv2dWithReluTest(%arg0: memref<1x16x4x4xf16>, %arg1: memref<1x16x3x3xf16>) -> memref<1x16x3x3xf16> {
    %0 = IERT.Constant memref<16x16x2x2xf16, #NHWC> = dense<0.0> : tensor<16x16x2x2xf32>
    %1 = memref.alloc() : memref<1x16x3x3xf16>
    %2 = memref.alloc() : memref<1x16x4x4xf16, #NHWC>
    %3 = IERT.Reorder inputs(%arg0 : memref<1x16x4x4xf16>) outputs(%2 : memref<1x16x4x4xf16, #NHWC>) -> memref<1x16x4x4xf16, #NHWC>
    %4 = memref.alloc() : memref<1x16x3x3xf16, #NHWC>
    %5 = memref.alloc() : memref<16x16x2x2xf16, #NHWC, "CMX_NN">
    %6 = IERT.Copy inputs(%0 : memref<16x16x2x2xf16, #NHWC>) outputs(%5 : memref<16x16x2x2xf16, #NHWC, "CMX_NN">) -> memref<16x16x2x2xf16, #NHWC, "CMX_NN">
    %7 = IERT.Constant memref<16x1x1x4xsi32> = dense<0> : tensor<16x1x1x4xsi32>
    %8 = memref.alloc() : memref<16x1x1x4xsi32, "CMX_NN">
    %9 = IERT.Copy inputs(%7 : memref<16x1x1x4xsi32>) outputs(%8 : memref<16x1x1x4xsi32, "CMX_NN">) -> memref<16x1x1x4xsi32, "CMX_NN">
    %10 = memref.alloc() : memref<1x16x4x4xf16, #NHWC, "CMX_NN">
    %11 = IERT.Copy inputs(%3 : memref<1x16x4x4xf16, #NHWC>) outputs(%10 : memref<1x16x4x4xf16, #NHWC, "CMX_NN">) -> memref<1x16x4x4xf16, #NHWC, "CMX_NN">
    %12 = memref.alloc() : memref<1x16x3x3xf16, #NHWC, "CMX_NN">
    %13 = VPUIP.NCEClusterTask {
            kernel_padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32],
            kernel_size = [2 : i32, 2 : i32],
            kernel_strides = [1 : i32, 1 : i32],
            task_type = "CONV"
        }
        input(%11 : memref<1x16x4x4xf16, #NHWC, "CMX_NN">)
        weights(%6 : memref<16x16x2x2xf16, #NHWC, "CMX_NN">)
        weight_table(%9 : memref<16x1x1x4xsi32, "CMX_NN">)
        parent_input(%11 : memref<1x16x4x4xf16, #NHWC, "CMX_NN">)
        parent_output(%12 : memref<1x16x3x3xf16, #NHWC, "CMX_NN">)
        outputs(%12 : memref<1x16x3x3xf16, #NHWC, "CMX_NN">)
        -> memref<1x16x3x3xf16, #NHWC, "CMX_NN">
        variants : {
            VPUIP.DPUTask {
                end = [2 : i32, 2 : i32, 15 : i32],
                mpe_mode = "VECTOR_FP16",
                pads_begin = [0 : i32, 0 : i32],
                pads_end = [0 : i32, 0 : i32],
                start = [0 : i32, 0 : i32, 0 : i32]
            }
        }
        PPE : {
        }
    %14 = IERT.Copy inputs(%13 : memref<1x16x3x3xf16, #NHWC, "CMX_NN">) outputs(%4 : memref<1x16x3x3xf16, #NHWC>) -> memref<1x16x3x3xf16, #NHWC>
    %15 = IERT.Reorder inputs(%14 : memref<1x16x3x3xf16, #NHWC>) outputs(%1 : memref<1x16x3x3xf16>) -> memref<1x16x3x3xf16>
    %16 = memref.alloc() : memref<1x16x3x3xf16>
    %17 = IERT.ReLU inputs(%15 : memref<1x16x3x3xf16>) outputs(%16 : memref<1x16x3x3xf16>) -> memref<1x16x3x3xf16>
    %18 = IERT.Copy inputs(%17 : memref<1x16x3x3xf16>) outputs(%arg1 : memref<1x16x3x3xf16>) -> memref<1x16x3x3xf16>
    return %18 : memref<1x16x3x3xf16>
}

// CHECK: VPUIP.NCEClusterTask
// CHECK:     PPE :  {
// CHECK:         VPUIP.PPETask "LRELU"

// CHECK-NOT: IERT.ReLU

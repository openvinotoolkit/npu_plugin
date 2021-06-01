// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=VPU3700" --convert-to-nce-ops %s | FileCheck %s

// CHECK-LABEL: @Conv2dTest
func @Conv2dTest(%arg0: memref<1x16x16x16xf16>, %arg1: memref<1x16x16x16xf16>) -> memref<1x16x16x16xf16> {
    %0 = IERT.Constant memref<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    %1 = IERT.Constant memref<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %2 = memref.alloc() : memref<1x16x16x16xf16>

    %3 = IERT.Convolution {
            dilations = [1 : i32, 1 : i32],
            pads_begin = [0 : i32, 0 : i32],
            pads_end = [0 : i32, 0 : i32],
            strides = [1 : i32, 1 : i32]
        }
        inputs(%arg0 : memref<1x16x16x16xf16>, %0 : memref<16x16x1x1xf16>, %1 : memref<1x16x1x1xf16>)
        outputs(%2 : memref<1x16x16x16xf16>) -> memref<1x16x16x16xf16>

    %4 = IERT.Copy inputs(%3 : memref<1x16x16x16xf16>) outputs(%arg1 : memref<1x16x16x16xf16>) -> memref<1x16x16x16xf16>
    return %4 : memref<1x16x16x16xf16>
}

// CHECK:       [[FILTER_CST:%.+]] = IERT.Constant memref<16x16x1x1xf16>

// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16>

// CHECK:       [[FILTER_NHWC_BUF:%.+]] = memref.alloc() : memref<16x16x1x1xf16, #NHWC>
// CHECK:       [[FILTER_NHWC:%.+]] = IERT.Reorder
// CHECK-SAME:      inputs([[FILTER_CST]] : memref<16x16x1x1xf16>)
// CHECK-SAME:      outputs([[FILTER_NHWC_BUF]] : memref<16x16x1x1xf16, #NHWC>)

// CHECK:       [[FILTER_CMX_BUF:%.+]] = memref.alloc() : memref<16x16x1x1xf16, #NHWC, "CMX_NN">
// CHECK:       [[FILTER_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_NHWC]] : memref<16x16x1x1xf16, #NHWC>)
// CHECK-SAME:      outputs([[FILTER_CMX_BUF]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">)

// CHECK:       [[WEIGHTS_TABLE:%.+]] = IERT.Constant memref<16x1x1x4xsi32>

// CHECK:       [[WEIGHTS_TABLE_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x4xsi32, "CMX_NN">
// CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>)
// CHECK-SAME:      outputs([[WEIGHTS_TABLE_CMX_BUF]] : memref<16x1x1x4xsi32, "CMX_NN">)

// CHECK:       [[INPUT_NHWC_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC>
// CHECK:       [[INPUT_NHWC:%.+]] = IERT.Reorder
// CHECK-SAME:      inputs(%arg0 : memref<1x16x16x16xf16>)
// CHECK-SAME:      outputs([[INPUT_NHWC_BUF]] : memref<1x16x16x16xf16, #NHWC>)

// CHECK:       [[INPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, "CMX_NN">
// CHECK:       [[INPUT_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[INPUT_NHWC]] : memref<1x16x16x16xf16, #NHWC>)
// CHECK-SAME:      outputs([[INPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, "CMX_NN">
// CHECK:       [[OUTPUT_CMX:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:          fixed_ppe_task = "NOOP"
// CHECK-SAME:          kernel_padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
// CHECK-SAME:          kernel_size = [1 : i32, 1 : i32]
// CHECK-SAME:          strides = [1 : i32, 1 : i32]
// CHECK-SAME:          task_type = "CONV"
// CHECK-SAME:      inputs(
// CHECK-SAME:          [[INPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">
// CHECK-SAME:          [[FILTER_CMX]] : memref<16x16x1x1xf16, #NHWC, "CMX_NN">
// CHECK-SAME:          [[WEIGHTS_TABLE_CMX]] : memref<16x1x1x4xsi32, "CMX_NN">
// CHECK-SAME:      parent_input([[INPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_output([[OUTPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      variants :
// CHECK:               VPUIP.DPUTask {
// CHECK-SAME:              end = [15 : i32, 15 : i32, 15 : i32]
// CHECK-SAME:              mpe_mode = "VECTOR_FP16"
// CHECK-SAME:              pads_begin = [0 : i32, 0 : i32]
// CHECK-SAME:              pads_end = [0 : i32, 0 : i32]
// CHECK-SAME:              start = [0 : i32, 0 : i32, 0 : i32]

// CHECK:       [[OUTPUT_NHWC_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC>
// CHECK:       [[OUTPUT_NHWC:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_NHWC_BUF]] : memref<1x16x16x16xf16, #NHWC>)

// CHECK:       [[OUTPUT:%.+]] = IERT.Reorder
// CHECK-SAME:      inputs([[OUTPUT_NHWC]] : memref<1x16x16x16xf16, #NHWC>)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x16x16xf16>)

// CHECK:       [[OUTPUT_COPY:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT]] : memref<1x16x16x16xf16>)
// CHECK-SAME:      outputs(%arg1 : memref<1x16x16x16xf16>)
// CHECK:       return [[OUTPUT_COPY]] : memref<1x16x16x16xf16>

// -----

// CHECK-LABEL: @MaxPoolTest
func @MaxPoolTest(%arg0: memref<1x16x1x4xf16>, %arg1: memref<1x16x1x4xf16>) -> memref<1x16x1x4xf16> {
    %0 = memref.alloc() : memref<1x16x1x4xf16>

    %1 = IERT.MaxPool {
            kernel_size = [1 : i32, 1 : i32],
            pads_begin = [0 : i32, 0 : i32],
            pads_end = [0 : i32, 0 : i32],
            strides = [1 : i32, 1 : i32]
        }
        inputs(%arg0 : memref<1x16x1x4xf16>)
        outputs(%0 : memref<1x16x1x4xf16>) -> memref<1x16x1x4xf16>

    %2 = IERT.Copy inputs(%1 : memref<1x16x1x4xf16>) outputs(%arg1 : memref<1x16x1x4xf16>) -> memref<1x16x1x4xf16>
    return %2 : memref<1x16x1x4xf16>
}

// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x1x4xf16>

// CHECK:       [[ACT_WINDOW_CST:%.+]] = IERT.Constant memref<16x1x1x16xui8>

// CHECK:       [[ACT_WINDOW_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x16xui8, "CMX_NN">
// CHECK:       [[ACT_WINDOW_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[ACT_WINDOW_CST]] : memref<16x1x1x16xui8>)
// CHECK-SAME:      outputs([[ACT_WINDOW_CMX_BUF]] : memref<16x1x1x16xui8, "CMX_NN">)

// CHECK:       [[WEIGHTS_TABLE:%.+]] = IERT.Constant memref<16x1x1x4xsi32>

// CHECK:       [[WEIGHTS_TABLE_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x4xsi32, "CMX_NN">
// CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>)
// CHECK-SAME:      outputs([[WEIGHTS_TABLE_CMX_BUF]] : memref<16x1x1x4xsi32, "CMX_NN">)

// CHECK:       [[INPUT_NHWC_BUF:%.+]] = memref.alloc() : memref<1x16x1x4xf16, #NHWC>
// CHECK:       [[INPUT_NHWC:%.+]] = IERT.Reorder
// CHECK-SAME:      inputs(%arg0 : memref<1x16x1x4xf16>)
// CHECK-SAME:      outputs([[INPUT_NHWC_BUF]] : memref<1x16x1x4xf16, #NHWC>)

// CHECK:       [[INPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x1x4xf16, #NHWC, "CMX_NN">
// CHECK:       [[INPUT_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[INPUT_NHWC]] : memref<1x16x1x4xf16, #NHWC>)
// CHECK-SAME:      outputs([[INPUT_CMX_BUF]] : memref<1x16x1x4xf16, #NHWC, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x1x4xf16, #NHWC, "CMX_NN">
// CHECK:       [[OUTPUT_CMX:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:          activation_window_channel_length = 4 : i32
// CHECK-SAME:          fixed_ppe_task = "NOOP"
// CHECK-SAME:          kernel_padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
// CHECK-SAME:          kernel_size = [1 : i32, 1 : i32]
// CHECK-SAME:          strides = [1 : i32, 1 : i32]
// CHECK-SAME:          task_type = "MAXPOOL"
// CHECK-SAME:      inputs(
// CHECK-SAME:          [[INPUT_CMX]] : memref<1x16x1x4xf16, #NHWC, "CMX_NN">
// CHECK-SAME:          [[WEIGHTS_TABLE_CMX]] : memref<16x1x1x4xsi32, "CMX_NN">
// CHECK-SAME:          [[ACT_WINDOW_CMX]] : memref<16x1x1x16xui8, "CMX_NN">
// CHECK-SAME:      parent_input([[INPUT_CMX]] : memref<1x16x1x4xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      parent_output([[OUTPUT_CMX_BUF]] : memref<1x16x1x4xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_CMX_BUF]] : memref<1x16x1x4xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      variants :
// CHECK:               VPUIP.DPUTask {
// CHECK-SAME:              end = [3 : i32, 0 : i32, 15 : i32]
// CHECK-SAME:              mpe_mode = "VECTOR_FP16"
// CHECK-SAME:              pads_begin = [0 : i32, 0 : i32]
// CHECK-SAME:              pads_end = [0 : i32, 0 : i32]
// CHECK-SAME:              start = [0 : i32, 0 : i32, 0 : i32]

// CHECK:       [[OUTPUT_NHWC_BUF:%.+]] = memref.alloc() : memref<1x16x1x4xf16, #NHWC>
// CHECK:       [[OUTPUT_NHWC:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_CMX]] : memref<1x16x1x4xf16, #NHWC, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_NHWC_BUF]] : memref<1x16x1x4xf16, #NHWC>)

// CHECK:       [[OUTPUT:%.+]] = IERT.Reorder
// CHECK-SAME:      inputs([[OUTPUT_NHWC]] : memref<1x16x1x4xf16, #NHWC>)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x1x4xf16>)

// CHECK:       [[OUTPUT_COPY:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT]] : memref<1x16x1x4xf16>)
// CHECK-SAME:      outputs(%arg1 : memref<1x16x1x4xf16>)
// CHECK:       return [[OUTPUT_COPY]] : memref<1x16x1x4xf16>

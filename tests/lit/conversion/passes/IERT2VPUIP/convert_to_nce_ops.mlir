// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=VPU3700" --convert-to-nce-ops %s | FileCheck %s


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 256 + d2 * 16 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 16 + d1 * 16 + d2 * 16 + d3)>

// CHECK-LABEL: @Conv2dTest
func @Conv2dTest(%arg0: memref<1x16x16x16xf16, #NHWC, #map0>, %arg1: memref<1x16x16x16xf16, #NHWC, #map0>) -> memref<1x16x16x16xf16, #NHWC, #map0> {
    %0 = IERT.Constant memref<16x16x1x1xf16, #NHWC, #map1> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    %1 = IERT.Constant memref<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %2 = memref.alloc() : memref<1x16x16x16xf16, #NHWC, #map0>

    %3 = IERT.Convolution {
            dilations = [1 : i32, 1 : i32],
            pads_begin = [0 : i32, 0 : i32],
            pads_end = [0 : i32, 0 : i32],
            strides = [1 : i32, 1 : i32]
        }
        inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, #map0>, %0 : memref<16x16x1x1xf16, #NHWC, #map1>, %1 : memref<1x16x1x1xf16>)
        outputs(%2 : memref<1x16x16x16xf16, #NHWC, #map0>) -> memref<1x16x16x16xf16, #NHWC, #map0>

    %4 = IERT.Copy inputs(%3 : memref<1x16x16x16xf16, #NHWC, #map0>) outputs(%arg1 : memref<1x16x16x16xf16, #NHWC, #map0>) -> memref<1x16x16x16xf16, #NHWC, #map0>
    return %4 : memref<1x16x16x16xf16, #NHWC, #map0>
}

// CHECK-DAG:   [[FILTER_CST:%.+]] = IERT.Constant memref<16x16x1x1xf16, #NHWC, #map1>
// CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = IERT.Constant memref<16x1x1x4xsi32> = dense<{{.*}}> : tensor<16x1x1x4xsi32>

// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, #map0>

// CHECK:       [[FILTER_CMX_BUF:%.+]] = memref.alloc() : memref<16x16x1x1xf16, #NHWC, #map1, "CMX_NN">
// CHECK:       [[FILTER_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_CST]] : memref<16x16x1x1xf16, #NHWC, #map1>)
// CHECK-SAME:      outputs([[FILTER_CMX_BUF]] : memref<16x16x1x1xf16, #NHWC, #map1, "CMX_NN">)

// CHECK:       [[WEIGHTS_TABLE_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x4xsi32, "CMX_NN">
// CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>)
// CHECK-SAME:      outputs([[WEIGHTS_TABLE_CMX_BUF]] : memref<16x1x1x4xsi32, "CMX_NN">)

// CHECK:       [[INPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, #map0, "CMX_NN">
// CHECK:       [[INPUT_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, #map0>)
// CHECK-SAME:      outputs([[INPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, #map0, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, #map0, "CMX_NN">
// CHECK:       [[OUTPUT_CMX:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:          kernel_padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
// CHECK-SAME:          kernel_size = [1 : i32, 1 : i32]
// CHECK-SAME:          strides = [1 : i32, 1 : i32]
// CHECK-SAME:          task_type = "CONV"
// CHECK-SAME:      input([[INPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, #map0, "CMX_NN">)
// CHECK-SAME:      weights([[FILTER_CMX]] : memref<16x16x1x1xf16, #NHWC, #map1, "CMX_NN">)
// CHECK-SAME:      weight_table([[WEIGHTS_TABLE_CMX]] : memref<16x1x1x4xsi32, "CMX_NN">)
// CHECK-SAME:      parent_input([[INPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, #map0, "CMX_NN">)
// CHECK-SAME:      parent_output([[OUTPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, #map0, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_CMX_BUF]] : memref<1x16x16x16xf16, #NHWC, #map0, "CMX_NN">)
// CHECK-SAME:      variants :
// CHECK:               VPUIP.DPUTask {end = [15 : i32, 2 : i32, 15 : i32], mpe_mode = "VECTOR_FP16", pads_begin = [0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32], start = [0 : i32, 0 : i32, 0 : i32]}
// CHECK:               VPUIP.DPUTask {end = [15 : i32, 5 : i32, 15 : i32], mpe_mode = "VECTOR_FP16", pads_begin = [0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32], start = [0 : i32, 3 : i32, 0 : i32]}
// CHECK:               VPUIP.DPUTask {end = [15 : i32, 8 : i32, 15 : i32], mpe_mode = "VECTOR_FP16", pads_begin = [0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32], start = [0 : i32, 6 : i32, 0 : i32]}
// CHECK:               VPUIP.DPUTask {end = [15 : i32, 11 : i32, 15 : i32], mpe_mode = "VECTOR_FP16", pads_begin = [0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32], start = [0 : i32, 9 : i32, 0 : i32]}
// CHECK:               VPUIP.DPUTask {end = [15 : i32, 15 : i32, 15 : i32], mpe_mode = "VECTOR_FP16", pads_begin = [0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32], start = [0 : i32, 12 : i32, 0 : i32]}

// CHECK:       [[OUTPUT:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_CMX]] : memref<1x16x16x16xf16, #NHWC, #map0, "CMX_NN">)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC, #map0>)

// CHECK:       [[OUTPUT_COPY:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT]] : memref<1x16x16x16xf16, #NHWC, #map0>)
// CHECK-SAME:      outputs(%arg1 : memref<1x16x16x16xf16, #NHWC, #map0>)
// CHECK:       return [[OUTPUT_COPY]] : memref<1x16x16x16xf16, #NHWC, #map0>


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2 * 16 + d3)>

// CHECK-LABEL: @MaxPoolTest
func @MaxPoolTest(%arg0: memref<1x16x1x4xf16, #NHWC, #map>, %arg1: memref<1x16x1x4xf16, #NHWC, #map>) -> memref<1x16x1x4xf16, #NHWC, #map> {
    %0 = memref.alloc() : memref<1x16x1x4xf16, #NHWC, #map>

    %1 = IERT.MaxPool {
            kernel_size = [1 : i32, 1 : i32],
            pads_begin = [0 : i32, 0 : i32],
            pads_end = [0 : i32, 0 : i32],
            strides = [1 : i32, 1 : i32]
        }
        inputs(%arg0 : memref<1x16x1x4xf16, #NHWC, #map>)
        outputs(%0 : memref<1x16x1x4xf16, #NHWC, #map>) -> memref<1x16x1x4xf16, #NHWC, #map>

    %2 = IERT.Copy inputs(%1 : memref<1x16x1x4xf16, #NHWC, #map>) outputs(%arg1 : memref<1x16x1x4xf16, #NHWC, #map>) -> memref<1x16x1x4xf16, #NHWC, #map>
    return %2 : memref<1x16x1x4xf16, #NHWC, #map>
}

// CHECK-DAG:   [[ACT_WINDOW_CST:%.+]] = IERT.Constant memref<16x1x1x16xui8>
// CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = IERT.Constant memref<16x1x1x4xsi32> = dense<{{.*}}> : tensor<16x1x1x4xsi32>
// CHECK:       [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x1x4xf16, #NHWC, #map>

// CHECK:       [[ACT_WINDOW_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x16xui8, "CMX_NN">
// CHECK:       [[ACT_WINDOW_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[ACT_WINDOW_CST]] : memref<16x1x1x16xui8>)
// CHECK-SAME:      outputs([[ACT_WINDOW_CMX_BUF]] : memref<16x1x1x16xui8, "CMX_NN">)

// CHECK:       [[WEIGHTS_TABLE_CMX_BUF:%.+]] = memref.alloc() : memref<16x1x1x4xsi32, "CMX_NN">
// CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>)
// CHECK-SAME:      outputs([[WEIGHTS_TABLE_CMX_BUF]] : memref<16x1x1x4xsi32, "CMX_NN">)

// CHECK:       [[INPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x1x4xf16, #NHWC, #map, "CMX_NN">
// CHECK:       [[INPUT_CMX:%.+]] = IERT.Copy
// CHECK-SAME:      inputs(%arg0 : memref<1x16x1x4xf16, #NHWC, #map>)
// CHECK-SAME:      outputs([[INPUT_CMX_BUF]] : memref<1x16x1x4xf16, #NHWC, #map, "CMX_NN">)

// CHECK:       [[OUTPUT_CMX_BUF:%.+]] = memref.alloc() : memref<1x16x1x4xf16, #NHWC, #map, "CMX_NN">
// CHECK:       [[OUTPUT_CMX:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:          activation_window_channel_length = 4 : i32
// CHECK-SAME:          kernel_padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
// CHECK-SAME:          kernel_size = [1 : i32, 1 : i32]
// CHECK-SAME:          strides = [1 : i32, 1 : i32]
// CHECK-SAME:          task_type = "MAXPOOL"
// CHECK-SAME:      input([[INPUT_CMX]] : memref<1x16x1x4xf16, #NHWC, #map, "CMX_NN">)
// CHECK-SAME:      weight_table([[WEIGHTS_TABLE_CMX]] : memref<16x1x1x4xsi32, "CMX_NN">)
// CHECK-SAME:      activation_window([[ACT_WINDOW_CMX]] : memref<16x1x1x16xui8, "CMX_NN">)
// CHECK-SAME:      parent_input([[INPUT_CMX]] : memref<1x16x1x4xf16, #NHWC, #map, "CMX_NN">)
// CHECK-SAME:      parent_output([[OUTPUT_CMX_BUF]] : memref<1x16x1x4xf16, #NHWC, #map, "CMX_NN">)
// CHECK-SAME:      outputs([[OUTPUT_CMX_BUF]] : memref<1x16x1x4xf16, #NHWC, #map, "CMX_NN">)
// CHECK-SAME:      variants :
// CHECK:               VPUIP.DPUTask {
// CHECK-SAME:              end = [3 : i32, 0 : i32, 15 : i32]
// CHECK-SAME:              mpe_mode = "VECTOR_FP16"
// CHECK-SAME:              pads_begin = [0 : i32, 0 : i32]
// CHECK-SAME:              pads_end = [0 : i32, 0 : i32]
// CHECK-SAME:              start = [0 : i32, 0 : i32, 0 : i32]

// CHECK:       [[OUTPUT:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_CMX]] : memref<1x16x1x4xf16, #NHWC, #map, "CMX_NN">)
// CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x1x4xf16, #NHWC, #map>)

// CHECK:       [[OUTPUT_COPY:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT]] : memref<1x16x1x4xf16, #NHWC, #map>)
// CHECK-SAME:      outputs(%arg1 : memref<1x16x1x4xf16, #NHWC, #map>)
// CHECK:       return [[OUTPUT_COPY]] : memref<1x16x1x4xf16, #NHWC, #map>

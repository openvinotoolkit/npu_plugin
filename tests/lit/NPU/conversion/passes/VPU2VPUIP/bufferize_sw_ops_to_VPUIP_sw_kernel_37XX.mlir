//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

// CHECK-LABEL:  func.func @ConvertFP32ToFP16
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x4x4xf32>)
func.func @ConvertFP32ToFP16(%input: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf16> {
    %output = VPU.Convert(%input) {dstElemType = f16} : tensor<1x3x4x4xf32> -> tensor<1x3x4x4xf16>
    return %output : tensor<1x3x4x4xf16>

    // CHECK-NOT: VPU.Convert
    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x4x4xf32, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x3x4x4xf32>) outputs([[INPUT_BUFFER_CMX]] : memref<1x3x4x4xf32, [@CMX_NN, 0]>) -> memref<1x3x4x4xf32, [@CMX_NN, 0]>
    // CHECK: [[CONVERT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x4x4xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Convert inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x3x4x4xf32, [@CMX_NN, 0]>) outputs([[CONVERT_BUFFER_CMX]] as {{[^:]+}}: memref<1x3x4x4xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x4x4xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x3x4x4xf32, [@CMX_NN, 0]>, memref<1x3x4x4xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x3x4x4xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x3x4x4xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x3x4x4xf16>) -> memref<1x3x4x4xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x3x4x4xf16>
}

// -----
// CHECK-LABEL:  func.func @ConvertFP16ToFP32UsingSW
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x4x4xf16>)
func.func @ConvertFP16ToFP32UsingSW(%input: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {
    %output = VPU.Convert(%input) {dstElemType = f32} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf32>
    return %output : tensor<1x3x4x4xf32>

    // CHECK-NOT: VPU.Convert
    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x4x4xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x3x4x4xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x3x4x4xf16, [@CMX_NN, 0]>) -> memref<1x3x4x4xf16, [@CMX_NN, 0]>
    // CHECK: [[CONVERT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x4x4xf32, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Convert inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x3x4x4xf16, [@CMX_NN, 0]>) outputs([[CONVERT_BUFFER_CMX]] as {{[^:]+}}: memref<1x3x4x4xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x4x4xf32, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x3x4x4xf16, [@CMX_NN, 0]>, memref<1x3x4x4xf32, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x3x4x4xf32>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x3x4x4xf32, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x3x4x4xf32>) -> memref<1x3x4x4xf32>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x3x4x4xf32>
}

// -----
// CHECK-LABEL:  func.func @NCEClusterTilingConvertFP16ToFP32
// CHECK-SAME:      ({{[^:]+}}: memref<1x3x4x4xf16>)
func.func @NCEClusterTilingConvertFP16ToFP32(%input: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {
    %output = VPU.NCE.ClusterTiling (%input as %arg0: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {
        %cvt = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf32>
        VPU.Yield %cvt
    }
    return %output : tensor<1x3x4x4xf32>
    // CHECK: [[INPUT_TENSOR:%.+]] = bufferization.to_tensor {{[^:]+}} : memref<1x3x4x4xf16>
    // CHECK: [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[INPUT_TENSOR]] as {{[^:]+}}: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {

    // CHECK:   [[SWKERNEL_OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x3x4x4xf32>

    // CHECK:   [[SWKERNEL_OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Convert inputs({{[^:]+}} as {{[^:]+}}: memref<1x3x4x4xf16>) outputs([[SWKERNEL_OUTPUT_BUFFER]] as {{[^:]+}}: memref<1x3x4x4xf32>) on tile 0 -> memref<1x3x4x4xf32>{
    // CHECK:     VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x3x4x4xf16>, memref<1x3x4x4xf32>
    // CHECK:   }

    // CHECK:   [[SWKERNEL_OUTPUT_TENSOR:%.+]] = bufferization.to_tensor [[SWKERNEL_OUTPUT]] : memref<1x3x4x4xf32>
    // CHECK:   VPU.Yield [[SWKERNEL_OUTPUT_TENSOR]]
    // CHECK: }
    // CHECK: [[OUTPUT_DDR:%.+]] = bufferization.to_memref [[OUTPUT]] : memref<1x3x4x4xf32>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x3x4x4xf32>
}

// -----
// CHECK-LABEL:  func.func @SingleSWLayer
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x1x1x1000xf16>)
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @SingleSWLayer(%input: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %output = VPU.SoftMax(%input) {axisInd = 3, padSize = 3} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %output: tensor<1x1x1x1000xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x1x1x1000xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK: [[SOFTMAX_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_SoftMax inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[SOFTMAX_BUFFER_CMX]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 3]}({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:  }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1x1x1000xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x1x1x1000xf16>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: module @VPU.SW  {
// CHECK-NEXT: func.func private @builtin_Sigmoid(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16", VPU.task_type = @COMPUTE}
// CHECK-NEXT: func.func private @builtin_SoftMax(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64) attributes {VPU.kernel_code = "singleShaveSoftmax.cpp", VPU.kernel_entry = "singleShaveSoftmax", VPU.task_type = @COMPUTE}
// CHECK-NEXT: func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

// CHECK-LABEL:  func.func @ThreeSWLayers
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x1x1x2000xf16>)
func.func @ThreeSWLayers(%input: tensor<1x1x1x2000xf16>) -> tensor<1x1x1x2000xf16> {
    %sftmax = VPU.SoftMax(%input) {axisInd = 3} : tensor<1x1x1x2000xf16> -> tensor<1x1x1x2000xf16>
    %sigmoid = VPU.Sigmoid(%sftmax) {axisInd = 3} : tensor<1x1x1x2000xf16> -> tensor<1x1x1x2000xf16>
    %output = VPU.SoftMax(%sigmoid) {axisInd = 3} : tensor<1x1x1x2000xf16> -> tensor<1x1x1x2000xf16>

    return %output : tensor<1x1x1x2000xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x1x1x2000xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>

    // CHECK: [[SOFTMAX1_SW_OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SOFTMAX1_SW_OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_SoftMax inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[SOFTMAX1_SW_OUTPUT_BUFFER]] as {{[^:]+}}: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0]}({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x2000xf16, [@CMX_NN, 0]>, memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[SOFTMAX1_SW_OUTPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1x1x2000xf16>
    // CHECK: [[SOFTMAX1_SW_OUTPUT_CMX:%.+]] = VPUIP.Copy inputs([[SOFTMAX1_SW_OUTPUT]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[SOFTMAX1_SW_OUTPUT_BUFFER_CMX]] : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>

    // CHECK: [[SIGMOID_SW_INPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SIGMOID_SW_INPUT_CMX:%.+]] = VPUIP.Copy inputs([[SOFTMAX1_SW_OUTPUT_CMX]] : memref<1x1x1x2000xf16>) outputs([[SIGMOID_SW_INPUT_BUFFER]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SIGMOID_SW_OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SIGMOID_SW_OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Sigmoid inputs([[SIGMOID_SW_INPUT_CMX]] as {{[^:]+}}: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[SIGMOID_SW_OUTPUT_BUFFER]] as {{[^:]+}}: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x2000xf16, [@CMX_NN, 0]>, memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[SIGMOID_SW_OUTPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1x1x2000xf16>
    // CHECK: [[SIGMOID_SW_OUTPUT_CMX:%.+]] = VPUIP.Copy inputs([[SIGMOID_SW_OUTPUT]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[SIGMOID_SW_OUTPUT_BUFFER_CMX]] : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>

    // CHECK: [[SOFTMAX2_SW_INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SOFTMAX2_SW_INPUT_CMX:%.+]] = VPUIP.Copy inputs([[SIGMOID_SW_OUTPUT_CMX]] : memref<1x1x1x2000xf16>) outputs([[SOFTMAX2_SW_INPUT_BUFFER_CMX]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SOFTMAX2_SW_OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SOFTMAX2_SW_OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_SoftMax inputs([[SOFTMAX2_SW_INPUT_CMX]] as {{[^:]+}}: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[SOFTMAX2_SW_OUTPUT_BUFFER]] as {{[^:]+}}: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0]}({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x2000xf16, [@CMX_NN, 0]>, memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[SOFTMAX2_SW_OUTPUT_BUFFER_DDR:%.+]] = memref.alloc() : memref<1x1x1x2000xf16>
    // CHECK: [[SOFTMAX2_SW_OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[SOFTMAX2_SW_OUTPUT]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[SOFTMAX2_SW_OUTPUT_BUFFER_DDR]] : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>
    // CHECK: return [[SOFTMAX2_SW_OUTPUT_DDR]] : memref<1x1x1x2000xf16>
}

// -----
// CHECK-LABEL:  func.func @ReduceMean
// CHECK-SAME:      ([[ARG0:%.+]]: memref<1x512x7x7xf16>, [[ARG1:%.+]]: memref<1x512x7xf16>)
func.func @ReduceMean(%input0: tensor<1x512x7x7xf16>, %input1: tensor<1x512x7xf16>) -> tensor<1x512x7xf16> {
    %output = VPU.ReduceMean(%input0) {axes_value = [2]} : tensor<1x512x7x7xf16> -> tensor<1x512x7xf16>
    return %output : tensor<1x512x7xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x512x7x7xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x512x7x7xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x512x7x7xf16, [@CMX_NN, 0]>) -> memref<1x512x7x7xf16, [@CMX_NN, 0]>
    // CHECK: [[REDUCEMEAN_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x512x7xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_ReduceMean inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x512x7x7xf16, [@CMX_NN, 0]>) outputs([[REDUCEMEAN_BUFFER_CMX]] as {{[^:]+}}: memref<1x512x7xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x512x7xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 1, [2]]}({{[^:]+}}, {{[^:]+}}) : memref<1x512x7x7xf16, [@CMX_NN, 0]>, memref<1x512x7xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x512x7xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x512x7xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x512x7xf16>) -> memref<1x512x7xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x512x7xf16>
}

// -----
// CHECK-LABEL:  func.func @InterpolateSWLayerWithUnnecessaryScalingAxes
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x128x1x1xf16>)
func.func @InterpolateSWLayerWithUnnecessaryScalingAxes(%input: tensor<1x128x1x1xf16>) -> tensor<1x128x32x32xf16> {
    %output = VPU.Interpolate(%input) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [0, 1, 2, 3], initial_input_dims_attr = [1, 128, 1, 1], initial_input_offset_attr = [0, 0, 0, 0], initial_output_dims_attr = [1, 128, 32, 32], initial_output_offset_attr = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 3.200000e+00, 3.200000e+00], sizes_attr = [1, 128, 32, 32], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x128x1x1xf16> -> tensor<1x128x32x32xf16>

    return %output : tensor<1x128x32x32xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x128x1x1xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[INTERPOLATE_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x128x32x32xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Interpolate inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x128x1x1xf16, [@CMX_NN, 0]>) outputs([[INTERPOLATE_BUFFER_CMX]] as {{[^:]+}}: memref<1x128x32x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x32x32xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1, 1, 128, 1], [32, 32, 128, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}({{[^:]+}}, {{[^:]+}}) : memref<1x128x1x1xf16, [@CMX_NN, 0]>, memref<1x128x32x32xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x128x32x32xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x128x32x32xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x128x32x32xf16>) -> memref<1x128x32x32xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x128x32x32xf16>
}

// -----
// Case A: SW Kernel's input and output buffers don't all fit in NNCMX. Try to work as much as possible with NNCMX.
// Input buffer is smaller so input buffer will be placed in DDR and output buffer will be placed in NNCMX.

// CHECK-LABEL:  func.func @SingleSWLayerTooLargeForCMXCaseA
// CHECK-SAME:      ({{[^:]+}}: memref<1x128x50x50xf16>)
func.func @SingleSWLayerTooLargeForCMXCaseA(%input: tensor<1x128x50x50xf16>) -> tensor<1x128x75x75xf16> {
    %output = VPU.Interpolate(%input) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [0, 1, 2, 3], initial_input_dims_attr = [1, 128, 50, 50], initial_input_offset_attr = [0, 0, 0, 0], initial_output_dims_attr = [1, 128, 75, 75], initial_output_offset_attr = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.50000e+00, 1.50000e+00], sizes_attr = [1, 128, 75, 75], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x128x50x50xf16> -> tensor<1x128x75x75xf16>
    return %output : tensor<1x128x75x75xf16>

    // CHECK: [[OUTPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x128x75x75xf16, [@CMX_NN, 0]>

    // CHECK-NOT: VPUIP.Copy
    // CHECK-NOT: memref.alloc()

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Interpolate inputs({{[^:]+}} as {{[^:]+}}: memref<1x128x50x50xf16>) outputs([[OUTPUT_BUFFER_CMX]] as {{[^:]+}}: memref<1x128x75x75xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x75x75xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [50, 50, 128, 1], [75, 75, 128, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}({{[^:]+}}, {{[^:]+}}) : memref<1x128x50x50xf16>, memref<1x128x75x75xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x128x75x75xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x128x75x75xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x128x75x75xf16>) -> memref<1x128x75x75xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x128x75x75xf16>
}

// -----
// Case B: SW Kernel's input and output buffers don't all fit in NNCMX. Try to work as much as possible with NNCMX.
// Input buffer is larger so input buffer will be placed in NNCMX and output buffer will be placed in DDR.

// CHECK-LABEL:  func.func @SingleSWLayerTooLargeForCMXCaseB
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x128x75x75xf16>)
func.func @SingleSWLayerTooLargeForCMXCaseB(%input: tensor<1x128x75x75xf16>) -> tensor<1x128x50x50xf16> {
    %output = VPU.Interpolate(%input) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [0, 1, 2, 3], initial_input_dims_attr = [1, 128, 75, 75], initial_input_offset_attr = [0, 0, 0, 0], initial_output_dims_attr = [1, 128, 50, 50], initial_output_offset_attr = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 0.666666e+00, 0.666666e+00], sizes_attr = [1, 128, 50, 50], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x128x75x75xf16> -> tensor<1x128x50x50xf16>
    return %output : tensor<1x128x50x50xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x128x75x75xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x128x75x75xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x128x75x75xf16, [@CMX_NN, 0]>) -> memref<1x128x75x75xf16, [@CMX_NN, 0]>
    // CHECK: [[INTERPOLATE_BUFFER_DDR:%.+]] = memref.alloc() : memref<1x128x50x50xf16>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Interpolate inputs({{[^:]+}} as {{[^:]+}}: memref<1x128x75x75xf16, [@CMX_NN, 0]>) outputs([[INTERPOLATE_BUFFER_DDR]] as {{[^:]+}}: memref<1x128x50x50xf16>) on tile 0 -> memref<1x128x50x50xf16>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [75, 75, 128, 1], [50, 50, 128, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}({{[^:]+}}, {{[^:]+}}) : memref<1x128x75x75xf16, [@CMX_NN, 0]>, memref<1x128x50x50xf16>
    // CHECK: }

    // CHECK-NOT: memref.alloc()
    // CHECK-NOT: VPUIP.Copy

    // CHECK: return [[OUTPUT]] : memref<1x128x50x50xf16>
}

// -----
// Case C: Neither of SW Kernel's input and output buffers fit in NNCMX.
// Both buffers will be placed in DDR.

// CHECK-LABEL:  func.func @SingleSWLayerTooLargeForCMXCaseC
// CHECK-SAME:      ({{[^:]+}}: memref<1x1x1x1000000xf16>)
func.func @SingleSWLayerTooLargeForCMXCaseC(%input: tensor<1x1x1x1000000xf16>) -> tensor<1x1x1x1000000xf16> {
    %output = VPU.SoftMax(%input) {axisInd = 3} : tensor<1x1x1x1000000xf16> -> tensor<1x1x1x1000000xf16>
    return %output: tensor<1x1x1x1000000xf16>

    // CHECK: [[INPUT_BUFFER_DDR:%.+]] = memref.alloc() : memref<1x1x1x1000000xf16>

    // CHECK-NOT: memref.alloc()
    // CHECK-NOT: VPUIP.Copy

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_SoftMax inputs({{[^:]+}} as {{[^:]+}}: memref<1x1x1x1000000xf16>) outputs([[INPUT_BUFFER_DDR]] as {{[^:]+}}: memref<1x1x1x1000000xf16>) on tile 0 -> memref<1x1x1x1000000xf16>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0]}({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x1000000xf16>, memref<1x1x1x1000000xf16>
    // CHECK: }

    // CHECK-NOT: memref.alloc()
    // CHECK-NOT: VPUIP.Copy

    // CHECK: return [[OUTPUT]] : memref<1x1x1x1000000xf16>
}

// -----
// Case D: SW Kernel's input and output buffers don't all fit in NNCMX. Try to work as much as possible with NNCMX.
// Both input buffers can fit together in NNCMX and together are larger than the output buffer.
// Both input buffers will be placed in NNCMX and the output buffer will be placed in DDR.

// CHECK-LABEL:  func.func @SingleSWLayerTooLargeForCMXCaseD
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x16x210x210xf16>)
func.func @SingleSWLayerTooLargeForCMXCaseD(%input: tensor<1x16x210x210xf16>) -> tensor<1x8x208x208xf16> {
    %cst = const.Declare tensor<8x8x3x3xf16> = dense<2.0> : tensor<2x4x8x3x3xf16>, [#const.Reshape<[8, 8, 3, 3]>]
    %output = VPU.GroupConvolution(%input, %cst) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x210x210xf16>, tensor<8x8x3x3xf16> -> tensor<1x8x208x208xf16>
    return %output : tensor<1x8x208x208xf16>

    // CHECK: [[CST_DECLARE:%.+]] = const.Declare tensor<8x8x3x3xf16> = dense<2.000000e+00> : tensor<2x4x8x3x3xf16>, [#const.Reshape<[8, 8, 3, 3]>]
    // CHECK: [[CST_DECLARE_BUFFER:%.+]] = bufferization.to_memref [[CST_DECLARE]] : memref<8x8x3x3xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x16x210x210xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x16x210x210xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x16x210x210xf16, [@CMX_NN, 0]>) -> memref<1x16x210x210xf16, [@CMX_NN, 0]>

    // CHECK: [[CST_DECLARE_BUFFER_CMX:%.+]] = memref.alloc() : memref<8x8x3x3xf16, [@CMX_NN, 0]>
    // CHECK: [[CST_DECLARE_CMX:%.+]] = VPUIP.Copy inputs([[CST_DECLARE_BUFFER]] : memref<8x8x3x3xf16>) outputs([[CST_DECLARE_BUFFER_CMX]] : memref<8x8x3x3xf16, [@CMX_NN, 0]>) -> memref<8x8x3x3xf16, [@CMX_NN, 0]>

    // CHECK: [[GROUPCONV_BUFFER:%.+]] = memref.alloc() : memref<1x8x208x208xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_GroupConvolution inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x16x210x210xf16, [@CMX_NN, 0]>, [[CST_DECLARE_CMX]] as {{[^:]+}}: memref<8x8x3x3xf16, [@CMX_NN, 0]>) outputs([[GROUPCONV_BUFFER]] as {{[^:]+}}: memref<1x8x208x208xf16>) on tile 0 -> memref<1x8x208x208xf16>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[1, 1], [0, 0], [0, 0], [1, 1], 2]}
    // CHECK:   ({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x16x210x210xf16, [@CMX_NN, 0]>, memref<8x8x3x3xf16, [@CMX_NN, 0]>, memref<1x8x208x208xf16>
    // CHECK: }

    // CHECK-NOT: memref.alloc()
    // CHECK-NOT: VPUIP.Copy

    // CHECK: return [[OUTPUT]] : memref<1x8x208x208xf16>
}

// -----
// CHECK-LABEL:  func.func @StridedSlice2Dim
// CHECK-SAME:      ([[ARG:%.+]]: memref<3x40x40x15xf16>)
func.func @StridedSlice2Dim(%input: tensor<3x40x40x15xf16>) -> tensor<3x40x20x5xf16> {
    %output = VPU.StridedSlice(%input) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 3]} : tensor<3x40x40x15xf16> -> tensor<3x40x20x5xf16>
    return %output : tensor<3x40x20x5xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<3x40x40x15xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<3x40x40x15xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<3x40x40x15xf16, [@CMX_NN, 0]>) -> memref<3x40x40x15xf16, [@CMX_NN, 0]>

    // CHECK: [[STRIDESLICE_BUFFER_CMX:%.+]] = memref.alloc() : memref<3x40x20x5xf16, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_StridedSlice inputs([[INPUT_CMX]] as {{[^:]+}}: memref<3x40x40x15xf16, [@CMX_NN, 0]>) outputs([[STRIDESLICE_BUFFER_CMX]] as {{[^:]+}}: memref<3x40x20x5xf16, [@CMX_NN, 0]>) on tile 0 -> memref<3x40x20x5xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[0, 0, 0, 0], [3, 40, 40, 15], [1, 1, 2, 3]]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}) : memref<3x40x40x15xf16, [@CMX_NN, 0]>, memref<3x40x20x5xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<3x40x20x5xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<3x40x20x5xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER:%.+]] : memref<3x40x20x5xf16>) -> memref<3x40x20x5xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<3x40x20x5xf16>
}

// -----
// CHECK-LABEL:  func.func @StridedSlice1Dim
// CHECK-SAME:      ([[ARG:%.+]]: memref<3x40x40x15xf16>)
func.func @StridedSlice1Dim(%input: tensor<3x40x40x15xf16>) -> tensor<3x40x40x5xf16> {
    %output = VPU.StridedSlice(%input) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 3]} : tensor<3x40x40x15xf16> -> tensor<3x40x40x5xf16>
    return %output : tensor<3x40x40x5xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<3x40x40x15xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<3x40x40x15xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<3x40x40x15xf16, [@CMX_NN, 0]>) -> memref<3x40x40x15xf16, [@CMX_NN, 0]>
    // CHECK: [[STRIDESLICE_BUFFER_CMX:%.+]] = memref.alloc() : memref<3x40x40x5xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_StridedSlice inputs([[INPUT_CMX]] as {{[^:]+}}: memref<3x40x40x15xf16, [@CMX_NN, 0]>) outputs([[STRIDESLICE_BUFFER_CMX]] as {{[^:]+}}: memref<3x40x40x5xf16, [@CMX_NN, 0]>) on tile 0 -> memref<3x40x40x5xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[0, 0, 0, 0], [3, 40, 40, 15], [1, 1, 1, 3]]}
    // CHECK:   ({{[^:]+}}, {{[^:]+}}) : memref<3x40x40x15xf16, [@CMX_NN, 0]>, memref<3x40x40x5xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<3x40x40x5xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<3x40x40x5xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER:%.+]] : memref<3x40x40x5xf16>) -> memref<3x40x40x5xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<3x40x40x5xf16>
}

// -----
// CHECK-LABEL:  func.func @Convolution
// CHECK-SAME:      ([[ARG0:%.+]]: memref<1x32x64x64xf16>, [[ARG1:%.+]]: memref<64x32x3x3xf16>)
func.func @Convolution(
        %input: tensor<1x32x64x64xf16>,
        %filter: tensor<64x32x3x3xf16>)
        -> tensor<1x64x62x62xf16> {
    %output = VPU.Convolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16>, tensor<64x32x3x3xf16> -> tensor<1x64x62x62xf16>
    return %output : tensor<1x64x62x62xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x32x64x64xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x32x64x64xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x32x64x64xf16, [@CMX_NN, 0]>) -> memref<1x32x64x64xf16, [@CMX_NN, 0]>
    // CHECK: [[FILTER_BUFFER_CMX:%.+]] = memref.alloc() : memref<64x32x3x3xf16, [@CMX_NN, 0]>
    // CHECK: [[FILTER_CMX:%.+]] = VPUIP.Copy inputs([[ARG1]] : memref<64x32x3x3xf16>) outputs([[FILTER_BUFFER_CMX]] : memref<64x32x3x3xf16, [@CMX_NN, 0]>) -> memref<64x32x3x3xf16, [@CMX_NN, 0]>
    // CHECK: [[CONV_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x64x62x62xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Convolution inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x32x64x64xf16, [@CMX_NN, 0]>, [[FILTER_CMX]] as {{[^:]+}}: memref<64x32x3x3xf16, [@CMX_NN, 0]>) outputs([[CONV_BUFFER_CMX]] as {{[^:]+}}: memref<1x64x62x62xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x64x62x62xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[1, 1], [0, 0], [0, 0], [1, 1], 1]}
    // CHECK:   ({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x32x64x64xf16, [@CMX_NN, 0]>, memref<64x32x3x3xf16, [@CMX_NN, 0]>, memref<1x64x62x62xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x64x62x62xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x64x62x62xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x64x62x62xf16>) -> memref<1x64x62x62xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x64x62x62xf16>
}

// -----
// Neither of SW Kernel's input and output buffers fit in NNCMX, so both of them should be placed in DDR
// but they will later be converted from SW Kernel to VPUIP.PermuteDMA operations.
// Leave input and output buffers in NNCMX to not add a performance hit for DMA for working with DDR.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:  func.func @MemPermute
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x3x1024x1024xf16>)
func.func @MemPermute(%input: tensor<1x3x1024x1024xf16, {order = #NCHW}>) -> tensor<1x1024x3x1024xf16, {order = #NHWC}> {
    %memPermute = VPU.MemPermute(%input) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x1024x1024xf16, {order = #NCHW}> -> tensor<1x1024x3x1024xf16, {order = #NHWC}>
    return %memPermute: tensor<1x1024x3x1024xf16, {order = #NHWC}>

    // CHECK: [[MEMPERMUTE_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x3x1024x1024xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>) -> memref<1x3x1024x1024xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_MemPermute inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x3x1024x1024xf16, [@CMX_NN, 0]>) outputs([[MEMPERMUTE_BUFFER_CMX]] as {{[^:]+}}: memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[0, 1, 2, 3]]}
    // CHECK:   ({{[^:]+}}, {{[^:]+}}) : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>, memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1024x3x1024xf16, #NHWC>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x1024x3x1024xf16, #NHWC>) -> memref<1x1024x3x1024xf16, #NHWC>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x1024x3x1024xf16, #NHWC>
}

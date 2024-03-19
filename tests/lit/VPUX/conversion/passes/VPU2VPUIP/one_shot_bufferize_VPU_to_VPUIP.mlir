//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

// CHECK-LABEL:  func.func @ConvertFP32ToFP16
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x4x4xf32>)
func.func @ConvertFP32ToFP16(%input: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf16> {
    %output = VPU.Convert(%input) {dstElemType = f16} : tensor<1x3x4x4xf32> -> tensor<1x3x4x4xf16>
    return %output : tensor<1x3x4x4xf16>

    // CHECK-NOT: VPU.Convert
    // CHECK: [[INPUT_BUFFER:%.+]] = bufferization.to_memref [[INPUT]] : memref<1x3x4x4xf32>
    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x4x4xf32, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[INPUT_BUFFER]] : memref<1x3x4x4xf32>) outputs([[INPUT_BUFFER_CMX]] : memref<1x3x4x4xf32, [@CMX_NN, 0]>) -> memref<1x3x4x4xf32, [@CMX_NN, 0]>
    // CHECK: [[CONVERT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x4x4xf16, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Convert inputs([[INPUT_CMX]] as %arg1: memref<1x3x4x4xf32, [@CMX_NN, 0]>) outputs([[CONVERT_BUFFER_CMX]] as %arg2: memref<1x3x4x4xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x4x4xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run(%arg1, %arg2) : memref<1x3x4x4xf32, [@CMX_NN, 0]>, memref<1x3x4x4xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x3x4x4xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x3x4x4xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x3x4x4xf16>) -> memref<1x3x4x4xf16>
    // CHECK: [[OUTPUT_TENSOR:%.+]] = bufferization.to_tensor [[OUTPUT_DDR]] : memref<1x3x4x4xf16>
    // CHECK: return [[OUTPUT_TENSOR]] : tensor<1x3x4x4xf16>
}

// -----
// CHECK-LABEL:  func.func @ConvertFP16ToFP32UsingSW
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x4x4xf16>)
func.func @ConvertFP16ToFP32UsingSW(%input: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {
    %output = VPU.Convert(%input) {dstElemType = f32} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf32>
    return %output : tensor<1x3x4x4xf32>

    // CHECK-NOT: VPU.Convert
    // CHECK: [[INPUT_BUFFER:%.+]] = bufferization.to_memref [[INPUT]] : memref<1x3x4x4xf16>
    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x4x4xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[INPUT_BUFFER]] : memref<1x3x4x4xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x3x4x4xf16, [@CMX_NN, 0]>) -> memref<1x3x4x4xf16, [@CMX_NN, 0]>
    // CHECK: [[CONVERT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x4x4xf32, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Convert inputs([[INPUT_CMX]] as %arg1: memref<1x3x4x4xf16, [@CMX_NN, 0]>) outputs([[CONVERT_BUFFER_CMX]] as %arg2: memref<1x3x4x4xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x4x4xf32, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run(%arg1, %arg2) : memref<1x3x4x4xf16, [@CMX_NN, 0]>, memref<1x3x4x4xf32, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x3x4x4xf32>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x3x4x4xf32, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x3x4x4xf32>) -> memref<1x3x4x4xf32>
    // CHECK: [[OUTPUT_TENSOR:%.+]] = bufferization.to_tensor [[OUTPUT_DDR]] : memref<1x3x4x4xf32>
    // CHECK: return [[OUTPUT_TENSOR]] : tensor<1x3x4x4xf32>
}

// -----
// CHECK-LABEL:  func.func @NCEClusterTilingConvertFP16ToFP32
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x3x4x4xf16>)
func.func @NCEClusterTilingConvertFP16ToFP32(%input: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {
    %output = VPU.NCE.ClusterTiling (%input as %arg0: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {
        %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf32>
        VPU.Yield %0
    }
    return %output : tensor<1x3x4x4xf32>

    // CHECK: [[OUTPUT:%.+]] = VPU.NCE.ClusterTiling ([[INPUT]] as %arg1: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {
    // CHECK:   [[SWKERNEL_INPUT_BUFFER:%.+]] = bufferization.to_memref %arg1 : memref<1x3x4x4xf16>
    // CHECK:   [[SWKERNEL_OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x3x4x4xf32>
    // CHECK:   [[SWKERNEL_OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_Convert inputs([[SWKERNEL_INPUT_BUFFER]] as %arg2: memref<1x3x4x4xf16>) outputs([[SWKERNEL_OUTPUT_BUFFER]] as %arg3: memref<1x3x4x4xf32>) on tile 0 -> memref<1x3x4x4xf32>{
    // CHECK:      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x3x4x4xf16>, memref<1x3x4x4xf32>
    // CHECK:   }
    // CHECK:   [[SWKERNEL_OUTPUT_TENSOR:%.+]] = bufferization.to_tensor [[SWKERNEL_OUTPUT]] : memref<1x3x4x4xf32>
    // CHECK:   VPU.Yield [[SWKERNEL_OUTPUT_TENSOR]]
    // CHECK: }
    // CHECK: return [[OUTPUT]] : tensor<1x3x4x4xf32>
}

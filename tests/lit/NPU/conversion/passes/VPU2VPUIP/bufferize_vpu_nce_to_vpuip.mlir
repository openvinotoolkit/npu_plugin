//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceConv
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: memref<1x16x16x16xf16, #NHWC, @CMX_NN>
// CHECK-SAME: [[ARG1:%.+]]: memref<16x16x1x1xf16, #NHWC, @CMX_NN>
// CHECK-SAME: [[ARG2:%.+]]: memref<16x1x1x4xsi32, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>
func.func @NceConv(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                   %arg1: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                   %arg2: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>)
                   -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.Convolution(%arg0, %arg1, %arg2) {
                pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK-NOT: VPU.NCE.Convolution

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[ARG0]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights([[ARG1]] : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weight_table([[ARG2]] : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      parent_input([[ARG0]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)

    // CHECK: return
    // CHECK-SAME: memref<1x16x16x16xf16, #NHWC, @CMX_NN>
}

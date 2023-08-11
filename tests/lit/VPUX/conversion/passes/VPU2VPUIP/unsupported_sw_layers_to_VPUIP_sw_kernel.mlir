//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceHW" --convert-sw-layers-to-VPUIP-sw-kernel %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test {

// CHECK: module @VPU.SW {
// CHECK-NEXT:   func.func private @builtin_AdaptiveAvgPool(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy"}
// CHECK-NEXT:   func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

func.func @UnsupportedSWLayer(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x56x56xf16> {
    %cst = const.Declare tensor<2xsi32> = dense<[56, 56]> : tensor<2xsi32>
    %0 = VPU.AdaptiveAvgPool(%arg0, %cst) : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>
    return %0: tensor<1x32x56x56xf16>

// CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x32x112x112xf16> to memref<1x32x112x112xf16>
// CHECK-DAG: [[VAR0:%.*]] = const.Declare tensor<2xsi32> = dense<56> : tensor<2xsi32>
// CHECK: [[VAR1:%.*]] = builtin.unrealized_conversion_cast [[VAR0]] : tensor<2xsi32> to memref<2xsi32>
// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x32x112x112xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x32x112x112xf16>) outputs([[VAR2]] : memref<1x32x112x112xf16, [@CMX_NN, 0]>) -> memref<1x32x112x112xf16, [@CMX_NN, 0]>
// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
// CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR1]] : memref<2xsi32>) outputs([[VAR4]] : memref<2xsi32, [@CMX_NN, 0]>) -> memref<2xsi32, [@CMX_NN, 0]>
// CHECK: [[VAR6:%.*]] = memref.alloc() : memref<1x32x56x56xf16, [@CMX_NN, 0]>
// CHECK: [[VAR7:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_AdaptiveAvgPool inputs([[VAR3]] as %arg1: memref<1x32x112x112xf16, [@CMX_NN, 0]>, [[VAR5]] as %arg2: memref<2xsi32, [@CMX_NN, 0]>) outputs([[VAR6]] as %arg3: memref<1x32x56x56xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x56x56xf16, [@CMX_NN, 0]>{
// CHECK: VPUIP.SW.Kernel.run(%arg1, %arg2, %arg3) : memref<1x32x112x112xf16, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>, memref<1x32x56x56xf16, [@CMX_NN, 0]>
// CHECK: }
// CHECK: [[VAR8:%.*]] = memref.alloc() : memref<1x32x56x56xf16>
// CHECK: [[VAR9:%.*]] = VPUIP.Copy inputs([[VAR7]] : memref<1x32x56x56xf16, [@CMX_NN, 0]>) outputs([[VAR8]] : memref<1x32x56x56xf16>) -> memref<1x32x56x56xf16>
// CHECK: [[VAR10:%.*]] = builtin.unrealized_conversion_cast [[VAR9]] : memref<1x32x56x56xf16> to tensor<1x32x56x56xf16>
// CHECK: [[VAR10:%.*]] : tensor<1x32x56x56xf16>
}
}

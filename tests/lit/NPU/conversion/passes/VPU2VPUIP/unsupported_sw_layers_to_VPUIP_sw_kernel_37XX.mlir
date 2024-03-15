//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceHW" --convert-sw-layers-to-VPUIP-sw-kernel %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test {

// CHECK: module @VPU.SW {
// CHECK-NEXT:   func.func private @builtin_AdaptiveMaxPool(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy", VPU.task_type = @COMPUTE}
// CHECK-NEXT:   func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

func.func @UnsupportedSWLayer(%arg0: tensor<1x30x112x112xf16>) -> (tensor<1x30x56x56xf16>, tensor<1x30x56x56xsi32>) {
    %cst = const.Declare tensor<2xsi32> = dense<[56, 56]> : tensor<2xsi32>
    %0, %1 = VPU.AdaptiveMaxPool(%arg0, %cst) {index_element_type = si32} : tensor<1x30x112x112xf16>, tensor<2xsi32> -> tensor<1x30x56x56xf16>, tensor<1x30x56x56xsi32>
    return %0, %1: tensor<1x30x56x56xf16>,  tensor<1x30x56x56xsi32>

// CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x30x112x112xf16> to memref<1x30x112x112xf16>
// CHECK-DAG: [[VAR0:%.*]] = const.Declare tensor<2xsi32> = dense<56> : tensor<2xsi32>
// CHECK: [[VAR1:%.*]] = builtin.unrealized_conversion_cast [[VAR0]] : tensor<2xsi32> to memref<2xsi32>
// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x30x112x112xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x30x112x112xf16>) outputs([[VAR2]] : memref<1x30x112x112xf16, [@CMX_NN, 0]>) -> memref<1x30x112x112xf16, [@CMX_NN, 0]>
// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
// CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR1]] : memref<2xsi32>) outputs([[VAR4]] : memref<2xsi32, [@CMX_NN, 0]>) -> memref<2xsi32, [@CMX_NN, 0]>
// CHECK: [[VAR6:%.*]] = memref.alloc() : memref<1x30x56x56xf16, [@CMX_NN, 0]>
// CHECK: [[VAR7:%.*]] = memref.alloc() : memref<1x30x56x56xsi32, [@CMX_NN, 0]>
// CHECK: [[VAR8:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0>} @VPU.SW::@builtin_AdaptiveMaxPool inputs([[VAR3]] as %arg1: memref<1x30x112x112xf16, [@CMX_NN, 0]>, [[VAR5]] as %arg2: memref<2xsi32, [@CMX_NN, 0]>) outputs([[VAR6]] as %arg3: memref<1x30x56x56xf16, [@CMX_NN, 0]>, [[VAR7]] as %arg4: memref<1x30x56x56xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x30x56x56xf16, [@CMX_NN, 0]>, memref<1x30x56x56xsi32, [@CMX_NN, 0]>){
// CHECK: VPUIP.SW.Kernel.run(%arg1, %arg2, %arg3, %arg4) : memref<1x30x112x112xf16, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>, memref<1x30x56x56xf16, [@CMX_NN, 0]>, memref<1x30x56x56xsi32, [@CMX_NN, 0]>
// CHECK: }
// CHECK: [[VAR9:%.*]] = memref.alloc() : memref<1x30x56x56xf16>
// CHECK: [[VAR11:%.*]] = VPUIP.Copy inputs([[VAR8]]#0 : memref<1x30x56x56xf16, [@CMX_NN, 0]>) outputs([[VAR9]] : memref<1x30x56x56xf16>) -> memref<1x30x56x56xf16>
// CHECK: [[VAR10:%.*]] = memref.alloc() : memref<1x30x56x56xsi32>
// CHECK: [[VAR12:%.*]] = VPUIP.Copy inputs([[VAR8]]#1 : memref<1x30x56x56xsi32, [@CMX_NN, 0]>) outputs([[VAR10]] : memref<1x30x56x56xsi32>) -> memref<1x30x56x56xsi32>
// CHECK: [[VAR13:%.*]] = builtin.unrealized_conversion_cast [[VAR11]] : memref<1x30x56x56xf16> to tensor<1x30x56x56xf16>
// CHECK: [[VAR14:%.*]] = builtin.unrealized_conversion_cast [[VAR12]] : memref<1x30x56x56xsi32> to tensor<1x30x56x56xsi32>
// CHECK: return [[VAR13]], [[VAR14]] : tensor<1x30x56x56xf16>, tensor<1x30x56x56xsi32>

}
}

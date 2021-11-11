// RUN: vpux-opt --split-input-file --convert-sw-layers-to-VPUIP %s | FileCheck %s

module @Test attributes {VPUIP.arch = "MTL", VPUIP.compilationMode = "ReferenceHW"} {

func @SingleSWLayer(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = IERT.SoftMax {axisInd = 3} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    return %0: memref<1x1x1x1000xf16>


// CHECK: module @VPU.SW  {
// CHECK:     func private @builtin_SoftMax(memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>, i64) attributes {VPU.kernel_code = "softmax_fp16.cpp", VPU.kernel_entry = "softmax_fp16"}
// CHECK:   }
// CHECK:   func @SingleSWLayer(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
// CHECK:     %0 = memref.alloc() : memref<1x1x1x1000xf16, "CMX_NN">
// CHECK:     %1 = IERT.Copy inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%0 : memref<1x1x1x1000xf16, "CMX_NN">) -> memref<1x1x1x1000xf16, "CMX_NN">
// CHECK:     %2 = memref.alloc() : memref<1x1x1x1000xf16, "CMX_NN">
// CHECK:     %3 = VPUIP.SW.Kernel @VPU.SW::@builtin_SoftMax inputs(%1 : memref<1x1x1x1000xf16, "CMX_NN">) outputs(%2 : memref<1x1x1x1000xf16, "CMX_NN">) on tile 0 -> memref<1x1x1x1000xf16, "CMX_NN">  {
// CHECK:     ^bb0(%arg2: memref<1x1x1x1000xf16, "CMX_NN">, %arg3: memref<1x1x1x1000xf16, "CMX_NN">):
// CHECK:       %c3_i64 = arith.constant 3 : i64
// CHECK:       VPUIP.SW.Kernel.run(%arg2, %arg3, %c3_i64) : memref<1x1x1x1000xf16, "CMX_NN">, memref<1x1x1x1000xf16, "CMX_NN">, i64
// CHECK:     }
// CHECK:     %4 = IERT.Copy inputs(%3 : memref<1x1x1x1000xf16, "CMX_NN">) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
// CHECK:     return %4 : memref<1x1x1x1000xf16>
// CHECK:   }

}

}
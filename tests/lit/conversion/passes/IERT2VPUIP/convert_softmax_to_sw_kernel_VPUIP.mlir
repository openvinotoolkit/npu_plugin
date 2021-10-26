// RUN: vpux-opt --split-input-file --convert-sw-layers-to-VPUIP %s | FileCheck %s

module @Test attributes {VPUIP.arch = "MTL", VPUIP.compilationMode = "ReferenceHW"} {

// CHECK-LABEL: @SingleSWLayer
func @SingleSWLayer(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %0 = IERT.SoftMax {axisInd = 3} inputs(%arg0 : memref<1x1x1x1000xf16>) outputs(%arg1 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    return %0: memref<1x1x1x1000xf16>

    // CHECK: [[VAR0:%.*]] = VPUIP.SW.Kernel
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      inputs(%arg0 : memref<1x1x1x1000xf16>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x1x1x1000xf16>)

    // CHECK: return [[VAR0]] : memref<1x1x1x1000xf16>
}

}
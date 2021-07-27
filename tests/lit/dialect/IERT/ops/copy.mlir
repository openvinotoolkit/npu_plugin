// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @FoldCopy
func @FoldCopy(%arg0: memref<1x8x4x2xf16>, %arg1: memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16> {
    %0 = memref.alloc() : memref<1x8x4x2xf16>
    %1 = IERT.Copy inputs(%arg0 : memref<1x8x4x2xf16>) outputs(%0 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    %2 = memref.alloc() : memref<1x8x4x2xf16>
    %3 = IERT.SoftMax {axisInd = 1} inputs(%1 : memref<1x8x4x2xf16>) outputs(%2 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    %4 = memref.alloc() : memref<1x8x4x2xf16>
    %5 = IERT.Copy inputs(%3 : memref<1x8x4x2xf16>) outputs(%4 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    %6 = memref.alloc() : memref<1x8x4x2xf16>
    %7 = IERT.SoftMax {axisInd = 1} inputs(%5 : memref<1x8x4x2xf16>) outputs(%6 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    %8 = IERT.Copy inputs(%7 : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    return %8 : memref<1x8x4x2xf16>

    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR1:%.*]] = IERT.SoftMax {axisInd = 1 : i64} inputs(%arg0 : memref<1x8x4x2xf16>) outputs([[VAR0]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR3:%.*]] = IERT.SoftMax {axisInd = 1 : i64} inputs([[VAR1]] : memref<1x8x4x2xf16>) outputs([[VAR2]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR4:%.*]] = IERT.Copy inputs([[VAR3]] : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: return [[VAR4]] : memref<1x8x4x2xf16>
}

// -----

func @FuseCopies(%arg0: memref<1x8x4x2xf16>, %arg1: memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16> {
    %0 = memref.alloc() : memref<1x8x4x2xf16>
    %1 = IERT.SoftMax {axisInd = 1} inputs(%arg0 : memref<1x8x4x2xf16>) outputs(%0 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    %2 = memref.alloc() : memref<1x8x4x2xf16>
    %3 = IERT.Copy inputs(%1 : memref<1x8x4x2xf16>) outputs(%2 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    %4 = IERT.Copy inputs(%3 : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>

    return %4 : memref<1x8x4x2xf16>

    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR1:%.*]] = IERT.SoftMax {axisInd = 1 : i64} inputs(%arg0 : memref<1x8x4x2xf16>) outputs([[VAR0]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR2:%.*]] = IERT.Copy inputs([[VAR1]] : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: return [[VAR2]] : memref<1x8x4x2xf16>
}

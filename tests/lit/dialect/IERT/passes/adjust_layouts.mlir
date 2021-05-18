// RUN: vpux-opt --split-input-file --adjust-layouts %s | FileCheck %s

//
// The 'adjust-layouts' pass:
//
//   * Adds required layouts to memref
//

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

module @InOutNHCW {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x8x4x2xf16, #NHCW>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x8x4x2xf16, #NHCW>
    }

func @main(%arg0: memref<1x8x4x2xf16, #NHCW>, %arg1: memref<1x8x4x2xf16, #NHCW>) -> memref<1x8x4x2xf16, #NHCW> {
    %0 = memref.alloc() : memref<1x8x4x2xf16, #NHCW>
    %1 = IERT.GRN {bias = 1.0 : f32} inputs(%arg0 : memref<1x8x4x2xf16, #NHCW>) outputs(%0 : memref<1x8x4x2xf16, #NHCW>) -> memref<1x8x4x2xf16, #NHCW>
    %2 = memref.alloc() : memref<1x8x4x2xf16, #NHCW>
    %3 = IERT.SoftMax {axisInd = 1 : i32} inputs(%1 : memref<1x8x4x2xf16, #NHCW>) outputs(%2 : memref<1x8x4x2xf16, #NHCW>) -> memref<1x8x4x2xf16, #NHCW>
    %4 = memref.alloc() : memref<1x8x4x2xf16, #NHCW>
    %5 = IERT.GRN {bias = 1.0 : f32} inputs(%3 : memref<1x8x4x2xf16, #NHCW>) outputs(%arg1 : memref<1x8x4x2xf16, #NHCW>) -> memref<1x8x4x2xf16, #NHCW>
    return %5 : memref<1x8x4x2xf16, #NHCW>

    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x8x4x2xf16, #NHCW>
    // CHECK: [[VAR1:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR2:%.*]] = IERT.Reorder inputs(%arg0 : memref<1x8x4x2xf16, #NHCW>) outputs([[VAR1]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR3:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR4:%.*]] = IERT.GRN {bias = 1.000000e+00 : f32} inputs([[VAR2]] : memref<1x8x4x2xf16>) outputs([[VAR3]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR5:%.*]] = IERT.Reorder inputs([[VAR4]] : memref<1x8x4x2xf16>) outputs([[VAR0]] : memref<1x8x4x2xf16, #NHCW>) -> memref<1x8x4x2xf16, #NHCW>
    // CHECK: [[VAR6:%.*]] = memref.alloc() : memref<1x8x4x2xf16, #NHCW>
    // CHECK: [[VAR7:%.*]] = IERT.SoftMax {axisInd = 1 : i32} inputs([[VAR5]] : memref<1x8x4x2xf16, #NHCW>) outputs([[VAR6]] : memref<1x8x4x2xf16, #NHCW>) -> memref<1x8x4x2xf16, #NHCW>
    // CHECK: [[VAR8:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR9:%.*]] = IERT.Reorder inputs([[VAR7]] : memref<1x8x4x2xf16, #NHCW>) outputs([[VAR8]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR10:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR11:%.*]] = IERT.GRN {bias = 1.000000e+00 : f32} inputs([[VAR9]] : memref<1x8x4x2xf16>) outputs([[VAR10]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR12:%.*]] = IERT.Reorder inputs([[VAR11]] : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16, #NHCW>) -> memref<1x8x4x2xf16, #NHCW>
    // CHECK: return [[VAR12]] : memref<1x8x4x2xf16, #NHCW>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @DifferentOrders {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x8x4x2xf16>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x8x4x2xf16, #NHWC>
    }

func @main(%arg0: memref<1x8x4x2xf16>, %arg1: memref<1x8x4x2xf16, #NHWC>) -> memref<1x8x4x2xf16, #NHWC> {
    %0 = IERT.GRN {bias = 1.000000e+00 : f32} inputs(%arg0 : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16, #NHWC>) -> memref<1x8x4x2xf16, #NHWC>

    return %0 : memref<1x8x4x2xf16, #NHWC>

    // CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x8x4x2xf16>
    // CHECK: [[VAR1:%.*]] = IERT.GRN {bias = 1.000000e+00 : f32} inputs(%arg0 : memref<1x8x4x2xf16>) outputs([[VAR0]] : memref<1x8x4x2xf16>) -> memref<1x8x4x2xf16>
    // CHECK: [[VAR2:%.*]] = IERT.Reorder inputs([[VAR1]] : memref<1x8x4x2xf16>) outputs(%arg1 : memref<1x8x4x2xf16, #NHWC>) -> memref<1x8x4x2xf16, #NHWC>
    // CHECK: return [[VAR2]] : memref<1x8x4x2xf16, #NHWC>
}

}

// RUN: vpux-opt --split-input-file --convert-scalar-to-tensor %s | FileCheck %s

// CHECK-LABEL: @ConvertScalarToTensor
func @ConvertScalarToTensor(%arg0: memref<18x8x72x64xf16>, %arg1: memref<8x72x64xf16>) -> memref<8x72x64xf16> {
    %cst = const.Declare memref<si32> = #const.Content<dense<1> : tensor<si32>>

    %0 = IERT.Gather {axis_value = 0 : i64, batch_dims = 0 : i64} inputs(%arg0 : memref<18x8x72x64xf16>, %cst : memref<si32>) outputs(%arg1 : memref<8x72x64xf16>) -> memref<8x72x64xf16>

    return %0 : memref<8x72x64xf16>

    // CHECK:       %[[VAL0:.*]] = IERT.GenericReshape  inputs(%cst : memref<si32>) -> memref<1xsi32>
    // CHECK:       %[[VAL1:.*]] = IERT.Gather {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      inputs(%arg0 : memref<18x8x72x64xf16>, %[[VAL0]] : memref<1xsi32>) outputs(%arg1 : memref<8x72x64xf16>) -> memref<8x72x64xf16>
    // CHECK:       return %[[VAL1]]
}

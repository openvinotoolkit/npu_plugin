// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --convert-shape-to-4d %s | FileCheck %s

//
// The 'convert-shape-to-4d' pass:
//
//   * Updates both Function bodies and Function prototypes.
//   * It shouldn't touch user types defined in `IE.CNNNetwork`.
//   * It should update types for `Constant` operation.
//

// CHECK-LABEL: @ConvertNDto4D
module @ConvertNDto4D {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: IE.DataInfo "data" : memref<1x1000xf32>
        IE.DataInfo "data" : memref<1x1000xf32>
    }
    outputsInfo : {
        // CHECK: IE.DataInfo "prob" : memref<1x1000xf32>
        IE.DataInfo "prob" : memref<1x1000xf32>
    }

// CHECK: func @main(%arg0: tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32>
func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    // CHECK:       %[[OUT:.*]] = IE.SoftMax(%arg0)
    // CHECK-SAME:      {axisInd = 3 : i32}
    // CHECK-SAME:      tensor<1x1x1x1000xf32> -> tensor<1x1x1x1000xf32>

    return %prob : tensor<1x1000xf32>
    // CHECK: return %[[OUT]] : tensor<1x1x1x1000xf32>
}

}

// -----

// CHECK-LABEL: @ConstantLayer
module @ConstantLayer {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo : {
        // CHECK: IE.DataInfo "output" : memref<2x2xf32>
        IE.DataInfo "output" : memref<2x2xf32>
    }

// CHECK: func @main() -> tensor<1x1x2x2xf32>
func @main() -> tensor<2x2xf32> {
    %0 = constant
        dense<[
                  [1.0, 2.0],
                  [3.0, 4.0]
        ]> : tensor<2x2xf32>
    // CHECK:       %[[OUT:.*]] = constant
    // CHECK-SAME:      tensor<1x1x2x2xf32>

    return %0 : tensor<2x2xf32>
    // CHECK: return %[[OUT]] : tensor<1x1x2x2xf32>
}

}

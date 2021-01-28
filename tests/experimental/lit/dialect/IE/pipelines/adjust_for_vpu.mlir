// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --adjust-for-vpu %s | FileCheck %s

//
// The 'adjust-for-vpu' pass calls:
//
//   * `convert-precision-to-fp16`
//

// CHECK-LABEL: @Test
module @Test {

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

// CHECK: func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16>
func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    // CHECK:       %[[OUT:.*]] = IE.SoftMax(%arg0)
    // CHECK-SAME:      {axisInd = 1 : i32}
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf16>

    return %prob : tensor<1x1000xf32>
    // CHECK: return %[[OUT]] : tensor<1x1000xf16>
}

}

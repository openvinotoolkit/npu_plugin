// RUN: vpux-opt --set-compile-params="vpu-arch=VPU3400_A0" --adjust-for-vpu %s | FileCheck %s

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

// CHECK: func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16>
func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    return %prob : tensor<1x1000xf32>

    // CHECK:       [[VAR0:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [1, 1000]}

    // CHECK:       [[VAR1:%.+]] = IE.SoftMax([[VAR0]])
    // CHECK-SAME:      {axisInd = 1 : i32}
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf16>

    // CHECK:       [[VAR2:%.+]] = IE.Reshape([[VAR1]]) {shape_value = [1, 1, 1, 1000]}

    // CHECK: return [[VAR2]] : tensor<1x1x1x1000xf16>
}

}

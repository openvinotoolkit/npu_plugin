// RUN: vpux-opt --set-compile-params="vpu-arch=KMB" --adjust-for-vpu %s | FileCheck %s

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x1000xf32>
        DataInfo "data" : tensor<1x1000xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x1000xf32>
        DataInfo "prob" : tensor<1x1000xf32>
    }

// CHECK: func @main([[ARG0:%.+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16>
func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    return %prob : tensor<1x1000xf32>

    // CHECK:       [[VAR1:%.+]] = IE.SoftMax([[ARG0]])
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf16>

    // CHECK: return [[VAR1]] : tensor<1x1000xf16>
}

}

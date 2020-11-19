// RUN: vpux-opt -split-input-file -adjust-precision-for-vpu %s | FileCheck %s

IE.CNNNetwork "FP32toFP16" at @main
    inputsInfo : {
        IE.DataInfo "data", f32, "NC"
    }
    outputsInfo : {
        IE.DataInfo "prob", f32, "NC"
    }

// CHECK: func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16>
func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    // CHECK:       %[[OUT:.*]] = IE.SoftMax(%arg0)
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf16>

    return %prob : tensor<1x1000xf32>
    // CHECK: return %[[OUT]] : tensor<1x1000xf16>
}

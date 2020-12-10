// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --adjust-precision-for-vpu %s | FileCheck %s

// CHECK-LABEL: FP32toFP16

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

// -----

// CHECK-LABEL: ConstantLayer

IE.CNNNetwork "ConstantLayer" at @main
    inputsInfo : {
    }
    outputsInfo : {
        IE.DataInfo "output", f32, "NCHW"
    }

// CHECK: func @main() -> tensor<1x2x2x2xf16>
func @main() -> tensor<1x2x2x2xf32> {
    %0 = constant
        dense<[
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0]
                ],
                [
                    [5.0, 6.0],
                    [7.0, 8.0]
                ]
            ]
        ]> : tensor<1x2x2x2xf32>
    // CHECK:       %[[OUT:.*]] = constant
    // CHECK-SAME:      tensor<1x2x2x2xf16>

    return %0 : tensor<1x2x2x2xf32>
    // CHECK: return %[[OUT]] : tensor<1x2x2x2xf16>
}

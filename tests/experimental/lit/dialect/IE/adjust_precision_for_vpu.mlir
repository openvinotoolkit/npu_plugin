// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --adjust-precision-for-vpu %s | FileCheck %s

// CHECK-LABEL: @FP32toFP16
module @FP32toFP16 {

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
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf16>

    return %prob : tensor<1x1000xf32>
    // CHECK: return %[[OUT]] : tensor<1x1000xf16>
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
        // CHECK: IE.DataInfo "output" : memref<1x2x2x2xf32>
        IE.DataInfo "output" : memref<1x2x2x2xf32>
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

}

// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --convert-Nd-ops-to-4d %s | FileCheck %s

// CHECK-LABEL: @ConvertNDto4D
module @ConvertNDto4D {

IE.CNNNetwork "ConvertNDto4D" at @main
    inputsInfo : {
        IE.DataInfo "data", f32, "NC"
    }
    outputsInfo : {
        IE.DataInfo "prob", f32, "NC"
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

IE.CNNNetwork "ConstantLayer" at @main
    inputsInfo : {
    }
    outputsInfo : {
        IE.DataInfo "output", f32, "NC"
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

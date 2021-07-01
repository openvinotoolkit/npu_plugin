// RUN: vpux-opt --split-input-file --use-user-precision %s | FileCheck %s

// CHECK-LABEL: @U8toFP32
module @U8toFP32 {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: IE.DataInfo "data" : tensor<1x1000xui8>
        IE.DataInfo "data" : tensor<1x1000xui8>
    }
    outputsInfo : {
        // CHECK: IE.DataInfo "prob" : tensor<1x1000xf32>
        IE.DataInfo "prob" : tensor<1x1000xf32>
    }

// CHECK: func @main(%[[ARG0:arg.*]]: tensor<1x1000xui8>) -> tensor<1x1000xf32>
func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK:       %[[VAR0:.*]] = IE.Convert(%[[ARG0]])
    // CHECK-SAME:      dstType = f16
    // CHECK-SAME:      tensor<1x1000xui8> -> tensor<1x1000xf16>

    // CHECK:       %[[VAR1:.*]] = IE.SoftMax(%[[VAR0]])
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf16>

    // CHECK:       %[[VAR2:.*]] = IE.Convert(%[[VAR1]])
    // CHECK-SAME:      dstType = f32
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf32>

    // CHECK:       return %[[VAR2]] : tensor<1x1000xf32>
}

}

// -----

// CHECK-LABEL: @SameTypes
module @SameTypes {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: IE.DataInfo "data" : tensor<1x1000xf16>
        IE.DataInfo "data" : tensor<1x1000xf16>
    }
    outputsInfo : {
        // CHECK: IE.DataInfo "prob" : tensor<1x1000xf16>
        IE.DataInfo "prob" : tensor<1x1000xf16>
    }

// CHECK: func @main(%[[ARG0:arg.*]]: tensor<1x1000xf16>) -> tensor<1x1000xf16>
func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK-NOT:   IE.Convert

    // CHECK:       %[[VAR0:.*]] = IE.SoftMax(%[[ARG0]])
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf16>

    // CHECK-NOT:   IE.Convert

    // CHECK:       return %[[VAR0]] : tensor<1x1000xf16>
}

}

// RUN: vpux-opt -split-input-file -reference-mode %s | FileCheck %s

// CHECK: #NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: VPUIP.Graph "SingleLayer" at @main
// CHECK-SAME:      options : "DynamicBarriers"
// CHECK-SAME:      item = "DDR", number = 0
// CHECK-SAME:      item = "SHAVE_UPA", number = 1
// CHECK-SAME:      item = "NCE_Cluster", number = 1
IE.CNNNetwork "SingleLayer" at @main
    inputsInfo : {
        // CHECK: VPUIP.TensorInfo "input", f32, #NC
        IE.DataInfo "input", f32, "NC"
    }
    outputsInfo : {
        // CHECK: VPUIP.TensorInfo "softmax", f32, #NC
        IE.DataInfo "softmax", f32, "NC"
    }

// CHECK:       func @main(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1000xf16>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1000xf16>) {
func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    return %0 : tensor<1x1000xf32>

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      isTrailingSWLayer
    // CHECK-SAME:      maxShaves = 1
    // CHECK-SAME:      inputs(%[[VAL_0]] : memref<1x1000xf16>)
    // CHECK-SAME:      outputs(%[[VAL_1]] : memref<1x1000xf16>)

    // CHECK:       return
}

// -----

// CHECK-LABEL: VPUIP.Graph "ConstantLayer" at @main
// CHECK-SAME:      options : "DynamicBarriers"
// CHECK-SAME:      item = "DDR", number = 0
// CHECK-SAME:      item = "SHAVE_UPA", number = 1
// CHECK-SAME:      item = "NCE_Cluster", number = 1
IE.CNNNetwork "ConstantLayer" at @main
    inputsInfo : {
        // CHECK: VPUIP.TensorInfo "input", f32, #NCHW
        IE.DataInfo "input", f32, "NCHW"
    }
    outputsInfo : {
        // CHECK: VPUIP.TensorInfo "output1", f32, #NCHW
        IE.DataInfo "output1", f32, "NCHW"
        // CHECK: VPUIP.TensorInfo "output2", f32, #NCHW
        IE.DataInfo "output2", f32, "NCHW"
    }

// CHECK:       func @main(
// CHECK-SAME:      %[[VAL_0:[a-z]*[a-z0-9]*]]: memref<1x2x2x2xf16>,
// CHECK-SAME:      %[[VAL_1:[a-z]*[a-z0-9]*]]: memref<1x2x2x2xf16>,
// CHECK-SAME:      %[[VAL_2:[a-z]*[a-z0-9]*]]: memref<1x2x2x2xf16>) {
func @main(%arg0: tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) {
    %cst = constant
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

    %0 = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x2x2x2xf32> -> tensor<1x2x2x2xf32>
    %1 = IE.SoftMax(%cst) {axisInd = 1 : i32} : tensor<1x2x2x2xf32> -> tensor<1x2x2x2xf32>

    return %0, %1 : tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>

    // CHECK:       %[[VAL_3:.*]] = VPUIP.DeclareConstantTensorOp
    // CHECK-SAME:      -> memref<1x2x2x2xf16>

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      isTrailingSWLayer
    // CHECK-SAME:      maxShaves = 1
    // CHECK-SAME:      inputs(%[[VAL_0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:      outputs(%[[VAL_1]] : memref<1x2x2x2xf16>)

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      isTrailingSWLayer
    // CHECK-SAME:      maxShaves = 1
    // CHECK-SAME:      inputs(%[[VAL_3]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:      outputs(%[[VAL_2]] : memref<1x2x2x2xf16>)

    // CHECK:       return
}

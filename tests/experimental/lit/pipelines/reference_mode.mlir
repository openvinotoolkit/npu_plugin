// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --reference-mode %s | FileCheck %s

// CHECK-LABEL: @SingleLayer
module @SingleLayer {

// CHECK:       VPUIP.Graph
// CHECK-SAME:      options : "DynamicBarriers"

// CHECK:   IERT.RunTimeResources
// CHECK:       usedMemory
// CHECK:           IERT.MemoryResource 0 bytes of "DDR"
// CHECK:       usedExecutors
// CHECK:           IERT.ExecutorResource 16 of "SHAVE_UPA"
// CHECK:           IERT.ExecutorResource 1 of "NCE_Cluster"

// CHECK: IE.CNNNetwork
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : memref<1x1000xf32>
    }
    outputsInfo : {
        IE.DataInfo "softmax" : memref<1x1000xf32>
    }

// CHECK:       func @main(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1x1x1000xf16>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1x1x1000xf16>) {
func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    return %0 : tensor<1x1000xf32>

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 3
    // CHECK-SAME:      isTrailingSWLayer
    // CHECK-SAME:      inputs(%[[VAL_0]] : memref<1x1x1x1000xf16>)
    // CHECK-SAME:      outputs(%[[VAL_1]] : memref<1x1x1x1000xf16>)

    // CHECK:       return
}

}

// -----

// CHECK-LABEL: @ConstantLayer
module @ConstantLayer {

// CHECK:       VPUIP.Graph
// CHECK-SAME:      options : "DynamicBarriers"

// CHECK:   IERT.RunTimeResources
// CHECK:       usedMemory
// CHECK:           IERT.MemoryResource 0 bytes of "DDR"
// CHECK:       usedExecutors
// CHECK:           IERT.ExecutorResource 16 of "SHAVE_UPA"
// CHECK:           IERT.ExecutorResource 1 of "NCE_Cluster"

// CHECK: IE.CNNNetwork
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : memref<1x2x2x2xf32>
    }
    outputsInfo : {
        IE.DataInfo "output1" : memref<1x2x2x2xf32>
        IE.DataInfo "output2" : memref<1x2x2x2xf32>
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
    // CHECK-SAME:      inputs(%[[VAL_0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:      outputs(%[[VAL_1]] : memref<1x2x2x2xf16>)

    // CHECK:       VPUIP.SoftMaxUPA
    // CHECK-SAME:      axisInd = 1
    // CHECK-SAME:      isTrailingSWLayer
    // CHECK-SAME:      inputs(%[[VAL_3]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:      outputs(%[[VAL_2]] : memref<1x2x2x2xf16>)

    // CHECK:       return
}

}

// -----

// CHECK-LABEL: @OptimizeUselessSoftMaxFP32
module @OptimizeUselessSoftMaxFP32 {

// CHECK:       VPUIP.Graph
// CHECK-SAME:      options : "DynamicBarriers"

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x2x4x2xf32>
    }

// CHECK:       func @main(
// CHECK-SAME:      %[[ARG:.*]]: memref<1x2x4x2xf16>
func @main() -> tensor<1x2x4x2xf32> {
    %0 = constant
        dense<[[
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                    [4.0, 5.0]
                ],
                [
                    [11.0, 22.0],
                    [22.0, 33.0],
                    [33.0, 44.0],
                    [44.0, 55.0]
                ]
        ]]> : tensor<1x2x4x2xf32>

    %prob = IE.SoftMax(%0) {axisInd = 0 : i32} : tensor<1x2x4x2xf32> -> tensor<1x2x4x2xf32>

    return %prob : tensor<1x2x4x2xf32>

    // CHECK:       %[[CST:.*]] = VPUIP.DeclareConstantTensorOp
    // CHECK-SAME:      -> memref<1x2x4x2xf16>

    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs(%[[CST]] : memref<1x2x4x2xf16>)
    // CHECK-SAME:      outputs(%[[ARG]] : memref<1x2x4x2xf16>)
}

}

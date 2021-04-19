// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=VPU3400_A0" --reference-mode %s | FileCheck %s

// CHECK-LABEL: @SingleLayer
module @SingleLayer {

// CHECK:       VPUIP.Graph
// CHECK-SAME:      options : "NONE"

// CHECK:   IERT.RunTimeResources
// CHECK:       usedMemory
// CHECK:           IERT.MemoryResource 2048 bytes of "DDR"

// CHECK: IE.CNNNetwork
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : memref<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "softmax" : memref<1x1000xf16>
    }

// CHECK:       func @main(
// CHECK-SAME:      [[ARG0:%.*]]: memref<1x1000xf16>,
// CHECK-SAME:      [[ARG1:%.*]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %0 : tensor<1x1000xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x1000xf16, "DDR">
    // CHECK:       [[VAR1:%.*]] = VPUIP.ConfigureBarrier<0> -> !VPUIP.Barrier
    // CHECK:       [[VAR2:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:              axisInd = 1
    // CHECK-SAME:              inputs([[ARG0]] : memref<1x1000xf16>)
    // CHECK-SAME:              outputs([[VAR0]] : memref<1x1000xf16, "DDR">)
    // CHECK-SAME:              updates([[VAR1]] : !VPUIP.Barrier)
    // CHECK:       [[VAR3:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:              inputs([[VAR2]] : memref<1x1000xf16, "DDR">)
    // CHECK-SAME:              outputs([[ARG1]] : memref<1x1000xf16>)
    // CHECK-SAME:              waits([[VAR1]] : !VPUIP.Barrier)

    // CHECK: return [[VAR3]] : memref<1x1000xf16>
}

}

// -----

// CHECK-LABEL: @ConstantLayer
module @ConstantLayer {

// CHECK:       VPUIP.Graph
// CHECK-SAME:      options : "NONE"

// CHECK:   IERT.RunTimeResources
// CHECK:       usedMemory
// CHECK:           IERT.MemoryResource 128 bytes of "DDR"

// CHECK: IE.CNNNetwork
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : memref<1x2x2x2xf16>
    }
    outputsInfo : {
        IE.DataInfo "output1" : memref<1x2x2x2xf16>
        IE.DataInfo "output2" : memref<1x2x2x2xf16>
    }

// CHECK:       func @main(
// CHECK-SAME:      [[ARG0:%.*]]: memref<1x2x2x2xf16>, [[ARG1:%.*]]: memref<1x2x2x2xf16>,
// CHECK-SAME:      [[ARG2:%.*]]: memref<1x2x2x2xf16>) -> (memref<1x2x2x2xf16>, memref<1x2x2x2xf16>) {
func @main(%arg0: tensor<1x2x2x2xf16>) -> (tensor<1x2x2x2xf16>, tensor<1x2x2x2xf16>) {
    %cst = IE.Constant tensor<1x2x2x2xf16> =
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

    %0 = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x2x2x2xf16> -> tensor<1x2x2x2xf16>
    %1 = IE.SoftMax(%cst) {axisInd = 1 : i32} : tensor<1x2x2x2xf16> -> tensor<1x2x2x2xf16>

    return %0, %1 : tensor<1x2x2x2xf16>, tensor<1x2x2x2xf16>

    // CHECK:   [[VAR0:%.*]] = VPUIP.DeclareConstantTensor
    // CHECK-SAME:              memref<1x2x2x2xf16>
    // CHECK:   [[VAR1:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <0> -> memref<1x2x2x2xf16, "DDR">
    // CHECK:   [[VAR2:%.*]] = VPUIP.ConfigureBarrier<0> -> !VPUIP.Barrier
    // CHECK:   [[VAR3:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:              axisInd = 1
    // CHECK-SAME:              inputs([[ARG0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:              outputs([[VAR1]] : memref<1x2x2x2xf16, "DDR">)
    // CHECK-SAME:              updates([[VAR2]] : !VPUIP.Barrier)

    // CHECK:   [[VAR4:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" <64> -> memref<1x2x2x2xf16, "DDR">
    // CHECK:   [[VAR5:%.*]] = VPUIP.ConfigureBarrier<1> -> !VPUIP.Barrier
    // CHECK:   [[VAR6:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:              axisInd = 1
    // CHECK-SAME:              inputs([[VAR0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:              outputs([[VAR4]] : memref<1x2x2x2xf16, "DDR">)
    // CHECK-SAME:              waits([[VAR2]] : !VPUIP.Barrier)
    // CHECK-SAME:              updates([[VAR5]] : !VPUIP.Barrier)

    // CHECK:   [[VAR7:%.*]] = VPUIP.ConfigureBarrier<0> -> !VPUIP.Barrier
    // CHECK:   [[VAR8:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:              inputs([[VAR3]] : memref<1x2x2x2xf16, "DDR">)
    // CHECK-SAME:              outputs([[ARG1]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:              waits([[VAR5]] : !VPUIP.Barrier)
    // CHECK-SAME:              updates([[VAR7]] : !VPUIP.Barrier)

    // CHECK:   [[VAR9:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:              inputs([[VAR6]] : memref<1x2x2x2xf16, "DDR">)
    // CHECK-SAME:              outputs([[ARG2]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:              waits([[VAR7]] : !VPUIP.Barrier)

    // CHECK:   return [[VAR8]], [[VAR9]] : memref<1x2x2x2xf16>, memref<1x2x2x2xf16>
}

}

// -----

// CHECK-LABEL: @OptimizeUselessSoftMaxFP32
module @OptimizeUselessSoftMaxFP32 {

// CHECK:       VPUIP.Graph
// CHECK-SAME:      options : "NONE"

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x2x4x2xf16>
    }

// CHECK:       func @main(
// CHECK-SAME:      %[[ARG:.*]]: memref<1x2x4x2xf16>) -> memref<1x2x4x2xf16> {
func @main() -> tensor<1x2x4x2xf16> {
    %0 = IE.Constant tensor<1x2x4x2xf16> =
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

    %prob = IE.SoftMax(%0) {axisInd = 0 : i32} : tensor<1x2x4x2xf16> -> tensor<1x2x4x2xf16>

    return %prob : tensor<1x2x4x2xf16>

    // CHECK:       %[[CST:.*]] = VPUIP.DeclareConstantTensor
    // CHECK-SAME:      memref<1x2x4x2xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs(%[[CST]] : memref<1x2x4x2xf16>)
    // CHECK-SAME:      outputs(%[[ARG]] : memref<1x2x4x2xf16>)

    // CHECK: return [[VAR0]] : memref<1x2x4x2xf16>
}

}

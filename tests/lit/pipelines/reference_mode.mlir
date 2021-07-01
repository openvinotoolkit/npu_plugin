// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=VPU3700" --reference-mode %s | FileCheck %s

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
        // CHECK: IE.DataInfo "input" : tensor<1x1000xf16>
        IE.DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        // CHECK: IE.DataInfo "softmax" : tensor<1x1000xf16>
        IE.DataInfo "softmax" : tensor<1x1000xf16>
    }

// CHECK:       func @main(
// CHECK-SAME:      [[ARG0:%.+]]: memref<1x1x1x1000xf16>,
// CHECK-SAME:      [[ARG1:%.+]]: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %0 : tensor<1x1000xf16>

    // CHECK-DAG:   [[VAR0:%.+]] = VPUIP.DeclareTensor "ProgrammableInput" [0] <0> -> memref<1x1000xf16>
    // CHECK-DAG:   [[VAR1:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x1000xf16, "DDR">
    // CHECK-DAG:   [[VAR2:%.+]] = VPUIP.ConfigureBarrier<0> -> !VPUIP.Barrier

    // CHECK-NEXT:  [[VAR3:%.+]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:              axisInd = 1
    // CHECK-SAME:              inputs([[VAR0]] : memref<1x1000xf16>)
    // CHECK-SAME:              outputs([[VAR1]] : memref<1x1000xf16, "DDR">)
    // CHECK-SAME:              updates([[VAR2]] : !VPUIP.Barrier)

    // CHECK-NEXT:  [[VAR4:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x1x1x1000xf16, "DDR">

    // CHECK-NEXT:  [[VAR5:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:              inputs([[VAR4]] : memref<1x1x1x1000xf16, "DDR">)
    // CHECK-SAME:              outputs([[ARG1]] : memref<1x1x1x1000xf16>)
    // CHECK-SAME:              waits([[VAR2]] : !VPUIP.Barrier)

    // CHECK-NEXT:  return [[VAR5]] : memref<1x1x1x1000xf16>
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
        IE.DataInfo "input" : tensor<1x2x2x2xf16>
    }
    outputsInfo : {
        IE.DataInfo "output1" : tensor<1x2x2x2xf16>
        IE.DataInfo "output2" : tensor<1x2x2x2xf16>
    }

// CHECK:       func @main(
// CHECK-SAME:      [[ARG0:%.+]]: memref<1x2x2x2xf16>, [[ARG1:%.+]]: memref<1x2x2x2xf16>,
// CHECK-SAME:      [[ARG2:%.+]]: memref<1x2x2x2xf16>) -> (memref<1x2x2x2xf16>, memref<1x2x2x2xf16>) {
func @main(%arg0: tensor<1x2x2x2xf16>) -> (tensor<1x2x2x2xf16>, tensor<1x2x2x2xf16>) {
    %cst = const.Declare tensor<1x2x2x2xf16> = #const.Content<
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
        ]> : tensor<1x2x2x2xf16>>

    %0 = IE.SoftMax(%arg0) {axisInd = 1 : i32} : tensor<1x2x2x2xf16> -> tensor<1x2x2x2xf16>
    %1 = IE.SoftMax(%cst) {axisInd = 1 : i32} : tensor<1x2x2x2xf16> -> tensor<1x2x2x2xf16>

    return %0, %1 : tensor<1x2x2x2xf16>, tensor<1x2x2x2xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare
    // CHECK-DAG:   [[BUF0:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x2x2x2xf16, "DDR">
    // CHECK-DAG:   [[BAR0:%.+]] = VPUIP.ConfigureBarrier<0> -> !VPUIP.Barrier

    // CHECK-NEXT:  [[VAR0:%.+]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:              axisInd = 1
    // CHECK-SAME:              inputs([[ARG0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:              outputs([[BUF0]] : memref<1x2x2x2xf16, "DDR">)
    // CHECK-SAME:              updates([[BAR0]] : !VPUIP.Barrier)

    // CHECK-DAG:   [[BUF1:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <64> -> memref<1x2x2x2xf16, "DDR">
    // CHECK-DAG:   [[BAR1:%.+]] = VPUIP.ConfigureBarrier<1> -> !VPUIP.Barrier

    // CHECK-NEXT:  [[VAR1:%.+]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:              axisInd = 1
    // CHECK-SAME:              inputs([[CST]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:              outputs([[BUF1]] : memref<1x2x2x2xf16, "DDR">)
    // CHECK-SAME:              waits([[BAR0]] : !VPUIP.Barrier)
    // CHECK-SAME:              updates([[BAR1]] : !VPUIP.Barrier)

    // CHECK-NEXT:  [[BAR2:%.+]] = VPUIP.ConfigureBarrier<2> -> !VPUIP.Barrier

    // CHECK-NEXT:  [[VAR2:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:              inputs([[VAR0]] : memref<1x2x2x2xf16, "DDR">)
    // CHECK-SAME:              outputs([[ARG1]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:              waits([[BAR1]] : !VPUIP.Barrier)
    // CHECK-SAME:              updates([[BAR2]] : !VPUIP.Barrier)

    // CHECK-NEXT:  [[VAR3:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:              inputs([[VAR1]] : memref<1x2x2x2xf16, "DDR">)
    // CHECK-SAME:              outputs([[ARG2]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:              waits([[BAR2]] : !VPUIP.Barrier)

    // CHECK-NEXT:  return [[VAR2]], [[VAR3]] : memref<1x2x2x2xf16>, memref<1x2x2x2xf16>
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
        IE.DataInfo "prob" : tensor<1x2x4x2xf16>
    }

// CHECK:       func @main(
// CHECK-SAME:      [[ARG:%.+]]: memref<1x2x4x2xf16>) -> memref<1x2x4x2xf16> {
func @main() -> tensor<1x2x4x2xf16> {
    %0 = const.Declare tensor<1x2x4x2xf16> = #const.Content<
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
        ]]> : tensor<1x2x4x2xf16>>

    %prob = IE.SoftMax(%0) {axisInd = 0 : i32} : tensor<1x2x4x2xf16> -> tensor<1x2x4x2xf16>

    return %prob : tensor<1x2x4x2xf16>

    // CHECK:       [[CST:%.+]] = const.Declare

    // CHECK-NEXT:  [[VAR0:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[CST]] : memref<1x2x4x2xf16>)
    // CHECK-SAME:      outputs([[ARG]] : memref<1x2x4x2xf16>)

    // CHECK-NEXT:  return [[VAR0]] : memref<1x2x4x2xf16>
}

}

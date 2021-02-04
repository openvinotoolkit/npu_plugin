// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=MA2490" --set-internal-memory-space="memory-space=DDR" %s | FileCheck %s

//
// The 'set-internal-memory-space' pass:
//
//   * Updates only Function bodies.
//   * Updates `alloc` Operation result Type.
//

// CHECK-LABEL: @Test
module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data" : memref<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x1000xf16>
    }

// CHECK: func @main([[ARG0:%arg[0-9]*]]: memref<1x1000xf16>, [[ARG1:%arg[0-9]*]]: memref<1x1000xf16>) {
func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) {
    %0 = alloc() : memref<1x1000xf16>
    IERT.SoftMax(%arg0, %0) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>

    %1 = alloc() : memref<1x1000xf16>
    IERT.SoftMax(%0, %1) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>

    IERT.SoftMax(%1, %arg1) {axisInd = 1 : i32} : memref<1x1000xf16>, memref<1x1000xf16>

    return

    // CHECK: [[VAR0:%[0-9]*]] = alloc() : memref<1x1000xf16, "DDR">
    // CHECK: IERT.SoftMax([[ARG0]], [[VAR0]])

    // CHECK: [[VAR1:%[0-9]*]] = alloc() : memref<1x1000xf16, "DDR">
    // CHECK: IERT.SoftMax([[VAR0]], [[VAR1]])

    // CHECK: IERT.SoftMax([[VAR1]], [[ARG1]])

    // CHECK: return
}

}

// -----

// CHECK-LABEL: @MultipleAllocs
module @MultipleAllocs {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "in" : memref<1x3x62x62xf32>
    }
    outputsInfo : {
        IE.DataInfo "out" : memref<1x48x30x30xf32>
    }

func @main(%in: memref<1x3x62x62xf32>, %out: memref<1x48x30x30xf32>) {
    // CHECK: IERT.Constant memref<48x3x3x3xf32>
    %cst0 = IERT.Constant memref<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>

    // CHECK: IERT.Constant memref<1x48x1x1xf32>
    %cst1 = IERT.Constant memref<1x48x1x1xf32> = dense<1.0> : tensor<1x48x1x1xf32>

    // CHECK: alloc() : memref<1x48x60x60xf32, "DDR">
    %temp0 = alloc() : memref<1x48x60x60xf32>

    // CHECK: memref<1x3x62x62xf32>, memref<48x3x3x3xf32>, memref<1x48x1x1xf32>, memref<1x48x60x60xf32, "DDR">
    IERT.Convolution(%in, %cst0, %cst1, %temp0)
        {
            dilations = [1 : i32, 1 : i32],
            pads_begin = [0 : i32, 0 : i32],
            pads_end = [0 : i32, 0 : i32],
            strides = [1 : i32, 1 : i32]
        } :
        memref<1x3x62x62xf32>, memref<48x3x3x3xf32>, memref<1x48x1x1xf32>, memref<1x48x60x60xf32>

    // CHECK: alloc() : memref<1x48x30x30xf32, "DDR">
    %temp1 = alloc() : memref<1x48x30x30xf32>

    // CHECK: memref<1x48x60x60xf32, "DDR">, memref<1x48x30x30xf32, "DDR">
    IERT.MaxPool(%temp0, %temp1)
        {
            kernel_size = [3 : i32, 3 : i32],
            pads_begin = [0 : i32, 0 : i32],
            pads_end = [0 : i32, 0 : i32],
            rounding_type = "CEIL",
            strides = [2 : i32, 2 : i32]
        } :
        memref<1x48x60x60xf32>, memref<1x48x30x30xf32>

    // CHECK: dealloc [[TEMP0:%[_a-z0-9]*]] : memref<1x48x60x60xf32, "DDR">
    dealloc %temp0 : memref<1x48x60x60xf32>

    // CHECK: memref<1x48x30x30xf32, "DDR">, memref<1x48x30x30xf32>
    IERT.ReLU(%temp1, %out) : memref<1x48x30x30xf32>, memref<1x48x30x30xf32>

    // CHECK: dealloc [[TEMP1:%[_a-z0-9]*]] : memref<1x48x30x30xf32, "DDR">
    dealloc %temp1 : memref<1x48x30x30xf32>

    return
}

}

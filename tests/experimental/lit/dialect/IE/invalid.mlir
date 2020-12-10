// RUN: vpux-opt %s --split-input-file --verify-diagnostics

// CHECK-LABEL: Wrong entryPoint

// expected-error@+1 {{'IE.CNNNetwork' entryPoint '@foo' doesn't refer to existing Function}}
IE.CNNNetwork "Test" at @foo
    inputsInfo : {
        IE.DataInfo "input", f16, "NHWC"
    }
    outputsInfo : {
        IE.DataInfo "softmax", f16, "NHWC"
    }

func @main(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x3x16x16xf32> {
    return %arg0 : tensor<1x3x16x16xf32>
}

// -----

// CHECK-LABEL: Wrong number of inputs

// expected-error@+1 {{'IE.CNNNetwork' entryPoint '@main' inputs count '2' doesn't match userInputs count '1'}}
IE.CNNNetwork "Test" at @main
    inputsInfo : {
        IE.DataInfo "input", f16, "NHWC"
    }
    outputsInfo : {
        IE.DataInfo "softmax", f16, "NHWC"
    }

func @main(%arg0: tensor<1x3x16x16xf32>, %arg1: tensor<1x3x16x16xf32>) -> tensor<1x3x16x16xf32> {
    return %arg0 : tensor<1x3x16x16xf32>
}

// -----

// CHECK-LABEL: Incompatible entry point signature

// expected-error@+1 {{'IE.CNNNetwork' entryPoint '@main' input #0 is not a 'RankedTensor'}}
IE.CNNNetwork "Test" at @main
    inputsInfo : {
        IE.DataInfo "input", f16, "NHWC"
    }
    outputsInfo : {
        IE.DataInfo "softmax", f16, "NHWC"
    }

func @main(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    return %arg0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: Incompatible input layout

// expected-error@+1 {{'IE.CNNNetwork' entryPoint '@main' input #0 is not compatible with userLayout 'NHWC'}}
IE.CNNNetwork "Test" at @main
    inputsInfo : {
        IE.DataInfo "input", f16, "NHWC"
    }
    outputsInfo : {
        IE.DataInfo "softmax", f16, "NHWC"
    }

func @main(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    return %arg0 : tensor<16xf32>
}

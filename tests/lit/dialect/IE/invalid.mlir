// RUN: vpux-opt %s --split-input-file --verify-diagnostics

// CHECK-LABEL: @wrong_entry_point
module @wrong_entry_point {

// expected-error@+1 {{entryPoint '@foo' doesn't refer to existing Function}}
IE.CNNNetwork
    entryPoint: @foo
    inputsInfo : {
        IE.DataInfo "input" : memref<1x3x16x16xf32>
    }
    outputsInfo : {
        IE.DataInfo "softmax" : memref<1x3x16x16xf32>
    }

func @main(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x3x16x16xf32> {
    return %arg0 : tensor<1x3x16x16xf32>
}

}

// -----

// CHECK-LABEL: @wrong_num_inputs
module @wrong_num_inputs {

// expected-error@+1 {{entryPoint '@main' has invalid state. inputs count '2', results count '1', user inputs count '1', user outputs count '1'}}
IE.CNNNetwork
    entryPoint: @main
    inputsInfo : {
        IE.DataInfo "input" : memref<1x3x16x16xf32>
    }
    outputsInfo : {
        IE.DataInfo "softmax" : memref<1x3x16x16xf32>
    }

func @main(%arg0: tensor<1x3x16x16xf32>, %arg1: tensor<1x3x16x16xf32>) -> tensor<1x3x16x16xf32> {
    return %arg0 : tensor<1x3x16x16xf32>
}

}

// -----

// CHECK-LABEL: @wrong_num_outputs
module @wrong_num_outputs {

// expected-error@+1 {{entryPoint '@main' outputs count '0' doesn't match userOutputs count '1'}}
IE.CNNNetwork
    entryPoint: @main
    inputsInfo : {
        IE.DataInfo "input" : memref<1x3x16x16xf32>
    }
    outputsInfo : {
        IE.DataInfo "softmax" : memref<1x3x16x16xf32>
    }

func @main(%arg0: tensor<1x3x16x16xf32>) {
    return
}

}
// -----

// CHECK-LABEL: @wrong_entry_point_sig
module @wrong_entry_point_sig {

// expected-error@+1 {{User input #0 is not a 'ShapedType'}}
IE.CNNNetwork
    entryPoint: @main
    inputsInfo : {
        IE.DataInfo "input" : f32
    }
    outputsInfo : {
        IE.DataInfo "softmax" : memref<1x3x16x16xf32>
    }

func @main(%arg0: memref<1x3x16x16xf32>) -> memref<1x3x16x16xf32> {
    return %arg0 : memref<1x3x16x16xf32>
}

}

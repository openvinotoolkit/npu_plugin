// RUN: vpux-opt %s --split-input-file --verify-diagnostics

// CHECK-LABEL: @wrong_entry_point
module @wrong_entry_point {

// expected-error@+1 {{entryPoint '@foo' doesn't refer to existing Function}}
IE.CNNNetwork
    entryPoint: @foo
    inputsInfo : {
        DataInfo "input" : tensor<1x3x16x16xf32>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x3x16x16xf32>
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
        DataInfo "input" : tensor<1x3x16x16xf32>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x3x16x16xf32>
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
        DataInfo "input" : tensor<1x3x16x16xf32>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x3x16x16xf32>
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
        DataInfo "input" : f16
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x3x16x16xf32>
    }

func @main(%arg0: memref<1x3x16x16xf32>) -> memref<1x3x16x16xf32> {
    return %arg0 : memref<1x3x16x16xf32>
}

}

// -----

// CHECK-LABEL: @wrong_tensor_attr
module @wrong_tensor_attr {

IE.CNNNetwork
    entryPoint: @main
    inputsInfo : {
        DataInfo "input" : tensor<16xf32>
    }
    outputsInfo : {
        DataInfo "output" : tensor<16xf32>
    }

func @main(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    return %arg0 : tensor<16xf32>
}

// expected-error@+1 {{Unsupported TensorType encoding '{qqq = "foo"}'}}
func private @extra(%arg0: tensor<16xf32, {qqq = "foo"}>)

}

// -----

// CHECK-LABEL: @wrong_tensor_attr_order1
module @wrong_tensor_attr_order1 {

IE.CNNNetwork
    entryPoint: @main
    inputsInfo : {
        DataInfo "input" : tensor<16xf32>
    }
    outputsInfo : {
        DataInfo "output" : tensor<16xf32>
    }

func @main(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    return %arg0 : tensor<16xf32>
}

// expected-error@+1 {{TensorType order '(d0, d1) -> (d0 * 10 + d1)' is not a permutation}}
func private @extra(%arg0: tensor<16xf32, {order = affine_map<(d0, d1) -> (d0 * 10 + d1)>}>)

}

// -----

// CHECK-LABEL: @wrong_tensor_attr_order2
module @wrong_tensor_attr_order2 {

IE.CNNNetwork
    entryPoint: @main
    inputsInfo : {
        DataInfo "input" : tensor<16xf32>
    }
    outputsInfo : {
        DataInfo "output" : tensor<16xf32>
    }

func @main(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    return %arg0 : tensor<16xf32>
}

// expected-error@+1 {{TensorType order '(d0, d1) -> (d1, d0)' doesn't match to shape '[16]'}}
func private @extra(%arg0: tensor<16xf32, {order = affine_map<(d0, d1) -> (d1, d0)>}>)

}

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

// expected-error@+1 {{entryPoint '@main' inputs count '2' doesn't match userInputs count '1'}}
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

// CHECK-LABEL: @wrong_entry_point_sig
module @wrong_entry_point_sig {

// expected-error@+1 {{entryPoint '@main' input #0 is not a 'ShapedType'}}
IE.CNNNetwork
    entryPoint: @main
    inputsInfo : {
        IE.DataInfo "input" : memref<1x3x16x16xf32>
    }
    outputsInfo : {
        IE.DataInfo "softmax" : memref<1x3x16x16xf32>
    }

func @main(%arg0: f32) -> f32 {
    return %arg0 : f32
}

}

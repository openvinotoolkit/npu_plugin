// RUN: vpux-opt --convert-shape-to-4d %s | FileCheck %s

// CHECK:       func @main(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1000xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<1x224x224xf32>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<1x512xf32>,
// CHECK-SAME:      %[[VAL_3:.*]]: tensor<8x1024xf32>
func @main(%arg0: tensor<1x1000xf32>, %arg1: tensor<1x224x224xf32>, %arg2: tensor<1x512xf32>, %arg3: tensor<8x1024xf32>) ->
        (tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>, tensor<8x1024xf32>) {
    %0 = IE.Clamp(%arg0) {min = 1.0 : f32, max = 3.0 : f32} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    %1 = IE.Sigmoid(%arg1) : tensor<1x224x224xf32> -> tensor<1x224x224xf32>
    %2 = IE.Elu(%1) {x = 1.0 : f32} : tensor<1x224x224xf32> -> tensor<1x224x224xf32>

    %input_low = IE.Constant tensor<1x1xf32> = dense<0.0> : tensor<1x1xf32>
    %input_high = IE.Constant tensor<1x1xf32> = dense<255.0> : tensor<1x1xf32>
    %output_low = IE.Constant tensor<1x1xf32> = dense<0.0> : tensor<1x1xf32>
    %output_high = IE.Constant tensor<1x1xf32> = dense<255.0> : tensor<1x1xf32>
    %3 = IE.FakeQuantize(%arg2, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 : i32 } :
        tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<1x512xf32>

    %4 = IE.Constant tensor<1xf32> = dense<6.0> : tensor<1xf32>
    %5 = IE.Constant tensor<1xf32> = dense<2.0> : tensor<1xf32>
    %6 = IE.Multiply(%arg3, %4) {auto_broadcast = "NUMPY"} : tensor<8x1024xf32>, tensor<1xf32> -> tensor<8x1024xf32>
    %7 = IE.Add(%6, %5) {auto_broadcast = "NUMPY"} : tensor<8x1024xf32>, tensor<1xf32> -> tensor<8x1024xf32>

    return %0, %2, %3, %7 : tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>, tensor<8x1024xf32>

    // CHECK-DAG: %[[VAL_4:.*]] = IE.Constant tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
    // CHECK-DAG: %[[VAL_5:.*]] = IE.Constant tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1xf32>
    // CHECK-DAG: %[[VAL_6:.*]] = IE.Constant tensor<1x1x1x1xf32> = dense<6.000000e+00> : tensor<1xf32>
    // CHECK-DAG: %[[VAL_7:.*]] = IE.Constant tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1xf32>

    // CHECK:   %[[VAL_8:.*]] = IE.Reshape(%[[VAL_0]]) {shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf32> -> tensor<1x1x1x1000xf32>
    // CHECK:   %[[VAL_9:.*]] = IE.Clamp(%[[VAL_8]])
    // CHECK:   %[[VAL_10:.*]] = IE.Reshape(%[[VAL_9]]) {shape_value = [1, 1000]} : tensor<1x1x1x1000xf32> -> tensor<1x1000xf32>

    // CHECK:   %[[VAL_11:.*]] = IE.Reshape(%[[VAL_1]]) {shape_value = [1, 1, 224, 224]} : tensor<1x224x224xf32> -> tensor<1x1x224x224xf32>
    // CHECK:   %[[VAL_12:.*]] = IE.Sigmoid(%[[VAL_11]])
    // CHECK:   %[[VAL_13:.*]] = IE.Elu(%[[VAL_12]])
    // CHECK:   %[[VAL_14:.*]] = IE.Reshape(%[[VAL_13]]) {shape_value = [1, 224, 224]} : tensor<1x1x224x224xf32> -> tensor<1x224x224xf32>

    // CHECK:   %[[VAL_15:.*]] = IE.Reshape(%[[VAL_2]]) {shape_value = [1, 1, 1, 512]} : tensor<1x512xf32> -> tensor<1x1x1x512xf32>
    // CHECK:   %[[VAL_16:.*]] = IE.FakeQuantize(%[[VAL_15]], %[[VAL_4]], %[[VAL_5]], %[[VAL_4]], %[[VAL_5]])
    // CHECK:   %[[VAL_17:.*]] = IE.Reshape(%[[VAL_16]]) {shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>

    // CHECK:   %[[VAL_18:.*]] = IE.Reshape(%[[VAL_3]]) {shape_value = [1, 1, 8, 1024]} : tensor<8x1024xf32> -> tensor<1x1x8x1024xf32>
    // CHECK:   %[[VAL_19:.*]] = IE.Multiply(%[[VAL_18]], %[[VAL_6]])
    // CHECK:   %[[VAL_20:.*]] = IE.Add(%[[VAL_19]], %[[VAL_7]])
    // CHECK:   %[[VAL_21:.*]] = IE.Reshape(%[[VAL_20]]) {shape_value = [8, 1024]} : tensor<1x1x8x1024xf32> -> tensor<8x1024xf32>

    // CHECK:   return %[[VAL_10]], %[[VAL_14]], %[[VAL_17]], %[[VAL_21]]
}

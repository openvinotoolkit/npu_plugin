// RUN: vpux-opt --split-input-file --convert-shape-to-4d --canonicalize %s | FileCheck %s

// CHECK:       func @main(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1x1x1000xf32>
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<1x1x224x224xf32>
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<1x1x1x512xf32>
// CHECK-SAME:      %[[VAL_3:.*]]: tensor<1x1x8x1024xf32>
func @main(%arg0: tensor<1x1000xf32>, %arg1: tensor<1x224x224xf32>, %arg2: tensor<1x512xf32>, %arg3: tensor<8x1024xf32>) ->
        (tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>, tensor<8x1024xf32>) {
    %0 = IE.Clamp(%arg0) {min = 1.0, max = 3.0} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    %1 = IE.Sigmoid(%arg1) : tensor<1x224x224xf32> -> tensor<1x224x224xf32>
    %2 = IE.Elu(%1) {x = 1.0} : tensor<1x224x224xf32> -> tensor<1x224x224xf32>

    %input_low = const.Declare tensor<1x1xf32> = #const.Content<dense<0.0> : tensor<1x1xf32>>
    %input_high = const.Declare tensor<1x1xf32> = #const.Content<dense<255.0> : tensor<1x1xf32>>
    %output_low = const.Declare tensor<1x1xf32> = #const.Content<dense<0.0> : tensor<1x1xf32>>
    %output_high = const.Declare tensor<1x1xf32> = #const.Content<dense<255.0> : tensor<1x1xf32>>
    %3 = IE.FakeQuantize(%arg2, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
        tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<1x512xf32>

    %4 = const.Declare tensor<1xf32> = #const.Content<dense<6.0> : tensor<1xf32>>
    %5 = const.Declare tensor<1xf32> = #const.Content<dense<2.0> : tensor<1xf32>>
    %6 = IE.Multiply(%arg3, %4) {auto_broadcast = "NUMPY"} : tensor<8x1024xf32>, tensor<1xf32> -> tensor<8x1024xf32>
    %7 = IE.Add(%6, %5) {auto_broadcast = "NUMPY"} : tensor<8x1024xf32>, tensor<1xf32> -> tensor<8x1024xf32>

    return %0, %2, %3, %7 : tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>, tensor<8x1024xf32>

    // CHECK-DAG: %[[VAL_4:.*]] = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<0.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>]>
    // CHECK-DAG: %[[VAL_5:.*]] = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<2.550000e+02> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>]>
    // CHECK-DAG: %[[VAL_6:.*]] = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<6.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>]>
    // CHECK-DAG: %[[VAL_7:.*]] = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<2.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>]>

    // CHECK:   %[[VAL_8:.*]] = IE.Clamp(%[[VAL_0]])
    // CHECK:   %[[VAL_9:.*]] = IE.Sigmoid(%[[VAL_1]])
    // CHECK:   %[[VAL_10:.*]] = IE.Elu(%[[VAL_9]])
    // CHECK:   %[[VAL_11:.*]] = IE.FakeQuantize(%[[VAL_2]], %[[VAL_4]], %[[VAL_5]], %[[VAL_4]], %[[VAL_5]])
    // CHECK:   %[[VAL_12:.*]] = IE.ScaleShift(%[[VAL_3]], %[[VAL_6]], %[[VAL_7]])
    // CHECK:   return %[[VAL_8]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]]
}

// -----

// CHECK:       func @FakeQuantizePerChannel3D(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1x512x64xf32>
func @FakeQuantizePerChannel3D(%arg0: tensor<1x512x64xf32>) -> (tensor<1x512x64xf32>) {
    %input_low = const.Declare tensor<1x512x1xf32> = #const.Content<dense<0.0> : tensor<1x512x1xf32>>
    %input_high = const.Declare tensor<1x512x1xf32> = #const.Content<dense<255.0> : tensor<1x512x1xf32>>
    %output_low = const.Declare tensor<1x512x1xf32> = #const.Content<dense<10.0> : tensor<1x512x1xf32>>
    %output_high = const.Declare tensor<1x512x1xf32> = #const.Content<dense<205.0> : tensor<1x512x1xf32>>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 : i32 } :
        tensor<1x512x64xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32> -> tensor<1x512x64xf32>

    return %3 : tensor<1x512x64xf32>

    // CHECK-DAG: %[[VAL_4:.*]] = const.Declare tensor<1x512x1x1xf32> = #const.Content<dense<2.050000e+02> : tensor<1x512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]>
    // CHECK-DAG: %[[VAL_3:.*]] = const.Declare tensor<1x512x1x1xf32> = #const.Content<dense<1.000000e+01> : tensor<1x512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]>
    // CHECK-DAG: %[[VAL_2:.*]] = const.Declare tensor<1x512x1x1xf32> = #const.Content<dense<2.550000e+02> : tensor<1x512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]>
    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x512x1x1xf32> = #const.Content<dense<0.000000e+00> : tensor<1x512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]>

    // CHECK:   %[[RESHAPE_BEFORE:.*]] = IE.Reshape(%[[VAL_0]]) {shape_value = [1, 512, 1, 64]} : tensor<1x1x512x64xf32> -> tensor<1x512x1x64xf32>
    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.*]] = IE.Reshape(%[[FQ]]) {shape_value = [1, 1, 512, 64]} : tensor<1x512x1x64xf32> -> tensor<1x1x512x64xf32>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK:       func @FakeQuantizePerChannel2D(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1x512x64xf32>
func @FakeQuantizePerChannel2D(%arg0: tensor<512x64xf32>) -> (tensor<512x64xf32>) {
    %input_low = const.Declare tensor<f32> = #const.Content<dense<0.0> : tensor<f32>>
    %input_high = const.Declare tensor<f32> = #const.Content<dense<255.0> : tensor<f32>>
    %output_low = const.Declare tensor<512x1xf32> = #const.Content<dense<10.0> : tensor<512x1xf32>>
    %output_high = const.Declare tensor<512x1xf32> = #const.Content<dense<205.0> : tensor<512x1xf32>>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 : i32 } :
        tensor<512x64xf32>, tensor<f32>, tensor<f32>, tensor<512x1xf32>, tensor<512x1xf32> -> tensor<512x64xf32>

    return %3 : tensor<512x64xf32>

    // CHECK-DAG: %[[VAL_4:.*]] = const.Declare tensor<1x512x1x1xf32> = #const.Content<dense<2.050000e+02> : tensor<512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]>
    // CHECK-DAG: %[[VAL_3:.*]] = const.Declare tensor<1x512x1x1xf32> = #const.Content<dense<1.000000e+01> : tensor<512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]>
    // CHECK-DAG: %[[VAL_2:.*]] = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<2.550000e+02> : tensor<f32>, [#const.Reshape<[1, 1, 1, 1]>]>
    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x1x1x1xf32> = #const.Content<dense<0.000000e+00> : tensor<f32>, [#const.Reshape<[1, 1, 1, 1]>]>

    // CHECK:   %[[RESHAPE_BEFORE:.*]] = IE.Reshape(%[[VAL_0]]) {shape_value = [1, 512, 64, 1]} : tensor<1x1x512x64xf32> -> tensor<1x512x64x1xf32>
    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.*]] = IE.Reshape(%[[FQ]]) {shape_value = [1, 1, 512, 64]} : tensor<1x512x64x1xf32> -> tensor<1x1x512x64xf32>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

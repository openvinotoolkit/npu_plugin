// RUN: vpux-opt --split-input-file --convert-shape-to-4d %s | FileCheck %s

// CHECK: [[MAP0:#.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: [[MAP1:#.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>

// CHECK-LABEL: @ConvertNDto4D
func @ConvertNDto4D(%arg0: tensor<1x1000xf32>,
                    %arg1: tensor<1x224x224xf32>,
                    %arg2: tensor<1x512xf32>,
                    %arg3: tensor<8x1024xf32>) ->
                    (tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>, tensor<8x1024xf32>) {
    // CHECK:           %[[VAL_0:.*]] = IE.Constant tensor<4xsi64> = dense<[1, 1, 1, 1000]> : tensor<4xsi64>
    // CHECK:           %[[VAL_1:.*]] = IE.Constant tensor<2xsi64> = dense<[1, 1000]> : tensor<2xsi64>
    // CHECK:           %[[VAL_2:.*]] = IE.Constant tensor<4xsi64> = dense<[1, 1, 224, 224]> : tensor<4xsi64>
    // CHECK:           %[[VAL_3:.*]] = IE.Constant tensor<3xsi64> = dense<[1, 224, 224]> : tensor<3xsi64>
    // CHECK:           %[[VAL_4:.*]] = IE.Constant tensor<4xsi64> = dense<[1, 1, 1, 512]> : tensor<4xsi64>
    // CHECK:           %[[VAL_5:.*]] = IE.Constant tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
    // CHECK:           %[[VAL_6:.*]] = IE.Constant tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1xf32>
    // CHECK:           %[[VAL_7:.*]] = IE.Constant tensor<2xsi64> = dense<[1, 512]> : tensor<2xsi64>
    // CHECK:           %[[VAL_8:.*]] = IE.Constant tensor<1x1x1x1xf32> = dense<6.000000e+00> : tensor<1xf32>
    // CHECK:           %[[VAL_9:.*]] = IE.Constant tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1xf32>
    // CHECK:           %[[VAL_10:.*]] = IE.Constant tensor<2xsi64> = dense<[8, 1024]> : tensor<2xsi64>

    %0 = IE.Clamp(%arg0) {min = 1.0 : f32, max = 3.0 : f32} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    // CHECK:           %[[VAL_11:.*]] = IE.Reshape(%arg0, %[[VAL_0]]) : tensor<1x1000xf32>, tensor<4xsi64> -> tensor<1x1x1x1000xf32>
    // CHECK:           %[[VAL_12:.*]] = IE.Clamp(%[[VAL_11]])
    // CHECK-SAME:          {max = 3.000000e+00 : f32, min = 1.000000e+00 : f32} :
    // CHECK-SAME:          tensor<1x1x1x1000xf32> -> tensor<1x1x1x1000xf32>
    // CHECK:           %[[VAL_13:.*]] = IE.Reshape(%[[VAL_12]], %[[VAL_1]]) : tensor<1x1x1x1000xf32>, tensor<2xsi64> -> tensor<1x1000xf32>

    %1 = IE.Sigmoid(%arg1) : tensor<1x224x224xf32> -> tensor<1x224x224xf32>
    %2 = IE.Elu(%1) {x = 1.0 : f32} : tensor<1x224x224xf32> -> tensor<1x224x224xf32>
    // CHECK:           %[[VAL_14:.*]] = IE.Reshape(%arg1, %[[VAL_2]]) : tensor<1x224x224xf32>, tensor<4xsi64> -> tensor<1x1x224x224xf32>
    // CHECK:           %[[VAL_15:.*]] = IE.Sigmoid(%[[VAL_14]]) : tensor<1x1x224x224xf32> -> tensor<1x1x224x224xf32>
    // CHECK:           %[[VAL_16:.*]] = IE.Elu(%[[VAL_15]])
    // CHECK-SAME:          {x = 1.000000e+00 : f32} :
    // CHECK-SAME:          tensor<1x1x224x224xf32> -> tensor<1x1x224x224xf32>
    // CHECK:           %[[VAL_17:.*]] = IE.Reshape(%[[VAL_16]], %[[VAL_3]]) : tensor<1x1x224x224xf32>, tensor<3xsi64> -> tensor<1x224x224xf32>

    %input_low = IE.Constant tensor<1x1xf32> = dense<0.0> : tensor<1x1xf32>
    %input_high = IE.Constant tensor<1x1xf32> = dense<255.0> : tensor<1x1xf32>
    %output_low = IE.Constant tensor<1x1xf32> = dense<0.0> : tensor<1x1xf32>
    %output_high = IE.Constant tensor<1x1xf32> = dense<255.0> : tensor<1x1xf32>

    %3 = IE.FakeQuantize(%arg2, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 : i32 } :
        tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<1x512xf32>
    // CHECK:           %[[VAL_18:.*]] = IE.Reshape(%arg2, %[[VAL_4]]) : tensor<1x512xf32>, tensor<4xsi64> -> tensor<1x1x1x512xf32>
    // CHECK:           %[[VAL_19:.*]] = IE.FakeQuantize(%[[VAL_18]], %[[VAL_5]], %[[VAL_6]], %[[VAL_5]], %[[VAL_6]])
    // CHECK-SAME:          {auto_broadcast = "NUMPY", levels = 256 : i32} :
    // CHECK-SAME:          tensor<1x1x1x512xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>,
    // CHECK-SAME:          tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x512xf32>
    // CHECK:           %[[VAL_20:.*]] = IE.Reshape(%[[VAL_19]], %[[VAL_7]]) : tensor<1x1x1x512xf32>, tensor<2xsi64> -> tensor<1x512xf32>

    %4 = IE.Constant tensor<1xf32> = dense<6.0> : tensor<1xf32>
    %5 = IE.Constant tensor<1xf32> = dense<2.0> : tensor<1xf32>
    %6 = IE.Multiply(%arg3, %4) {auto_broadcast = "NUMPY"}
        : tensor<8x1024xf32>, tensor<1xf32> -> tensor<8x1024xf32>
    %7 = IE.Add(%6, %5) {auto_broadcast = "NUMPY"}
        : tensor<8x1024xf32>, tensor<1xf32> -> tensor<8x1024xf32>
    // CHECK:           %[[VAL_21:.*]] = linalg.tensor_reshape %arg3 [#map0, #map1]
    // CHECK-SAME:          : tensor<8x1024xf32> into tensor<1x1x8x1024xf32>
    // CHECK:           %[[VAL_22:.*]] = IE.Multiply(%[[VAL_21]], %[[VAL_8]])
    // CHECK-SAME:          {auto_broadcast = "NUMPY"} :
    // CHECK-SAME:          tensor<1x1x8x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x8x1024xf32>
    // CHECK:           %[[VAL_23:.*]] = IE.Reshape(%[[VAL_22]], %[[VAL_10]])
    // CHECK-SAME:          : tensor<1x1x8x1024xf32>, tensor<2xsi64> -> tensor<8x1024xf32>
    // CHECK:           %[[VAL_24:.*]] = linalg.tensor_reshape %[[VAL_23]] [#map0, #map1]
    // CHECK-SAME:          : tensor<8x1024xf32> into tensor<1x1x8x1024xf32>
    // CHECK:           %[[VAL_25:.*]] = IE.Add(%[[VAL_24]], %[[VAL_9]])
    // CHECK-SAME:          {auto_broadcast = "NUMPY"} :
    // CHECK-SAME:          tensor<1x1x8x1024xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x8x1024xf32>
    // CHECK:           %[[VAL_26:.*]] = IE.Reshape(%[[VAL_25]], %[[VAL_10]]) :
    // CHECK-SAME:          tensor<1x1x8x1024xf32>, tensor<2xsi64> -> tensor<8x1024xf32>

    return %0, %2, %3, %7 : tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>, tensor<8x1024xf32>
    // CHECK:           return %[[VAL_13]], %[[VAL_17]], %[[VAL_20]], %[[VAL_26]]
    // CHECK-SAME:          : tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>, tensor<8x1024xf32>
}

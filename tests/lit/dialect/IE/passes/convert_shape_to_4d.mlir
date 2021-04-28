// RUN: vpux-opt --split-input-file --convert-shape-to-4d %s | FileCheck %s

// CHECK-LABEL: @ConvertNDto4D
func @ConvertNDto4D(%arg0: tensor<1x1000xf32>,
                    %arg1: tensor<1x224x224xf32>,
                    %arg2: tensor<1x512xf32>) ->
                    (tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>) {
    // CHECK:           %[[VAL_0:.*]] = IE.Constant tensor<4xsi64> = dense<[1, 1, 1, 1000]> : tensor<4xsi64>
    // CHECK:           %[[VAL_1:.*]] = IE.Constant tensor<2xsi64> = dense<[1, 1000]> : tensor<2xsi64>
    // CHECK:           %[[VAL_2:.*]] = IE.Constant tensor<4xsi64> = dense<[1, 1, 224, 224]> : tensor<4xsi64>
    // CHECK:           %[[VAL_3:.*]] = IE.Constant tensor<3xsi64> = dense<[1, 224, 224]> : tensor<3xsi64>
    // CHECK:           %[[VAL_4:.*]] = IE.Constant tensor<4xsi64> = dense<[1, 1, 1, 512]> : tensor<4xsi64>
    // CHECK:           %[[VAL_5:.*]] = IE.Constant tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>
    // CHECK:           %[[VAL_6:.*]] = IE.Constant tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1xf32>
    // CHECK:           %[[VAL_7:.*]] = IE.Constant tensor<2xsi64> = dense<[1, 512]> : tensor<2xsi64>

    %0 = IE.Clamp(%arg0) {min = 1.0 : f32, max = 3.0 : f32} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    // CHECK:           %[[VAL_8:.*]] = IE.Reshape(%arg0, %[[VAL_0]]) : tensor<1x1000xf32>, tensor<4xsi64> -> tensor<1x1x1x1000xf32>
    // CHECK:           %[[VAL_9:.*]] = IE.Clamp(%[[VAL_8]])
    // CHECK-SAME:          {max = 3.000000e+00 : f32, min = 1.000000e+00 : f32} :
    // CHECK-SAME:          tensor<1x1x1x1000xf32> -> tensor<1x1x1x1000xf32>
    // CHECK:           %[[VAL_10:.*]] = IE.Reshape(%[[VAL_9]], %[[VAL_1]]) : tensor<1x1x1x1000xf32>, tensor<2xsi64> -> tensor<1x1000xf32>

    %1 = IE.Sigmoid(%arg1) : tensor<1x224x224xf32> -> tensor<1x224x224xf32>
    %2 = IE.Elu(%1) {x = 1.0 : f32} : tensor<1x224x224xf32> -> tensor<1x224x224xf32>
    // CHECK:           %[[VAL_11:.*]] = IE.Reshape(%arg1, %[[VAL_2]]) : tensor<1x224x224xf32>, tensor<4xsi64> -> tensor<1x1x224x224xf32>
    // CHECK:           %[[VAL_12:.*]] = IE.Sigmoid(%[[VAL_11]]) : tensor<1x1x224x224xf32> -> tensor<1x1x224x224xf32>
    // CHECK:           %[[VAL_13:.*]] = IE.Elu(%[[VAL_12]])
    // CHECK-SAME:          {x = 1.000000e+00 : f32} :
    // CHECK-SAME:          tensor<1x1x224x224xf32> -> tensor<1x1x224x224xf32>
    // CHECK:           %[[VAL_14:.*]] = IE.Reshape(%[[VAL_13]], %[[VAL_3]]) : tensor<1x1x224x224xf32>, tensor<3xsi64> -> tensor<1x224x224xf32>

    %input_low = IE.Constant tensor<1x1xf32> = dense<0.0> : tensor<1x1xf32>
    %input_high = IE.Constant tensor<1x1xf32> = dense<255.0> : tensor<1x1xf32>
    %output_low = IE.Constant tensor<1x1xf32> = dense<0.0> : tensor<1x1xf32>
    %output_high = IE.Constant tensor<1x1xf32> = dense<255.0> : tensor<1x1xf32>

    %3 = IE.FakeQuantize(%arg2, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 : i32 } :
        tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<1x512xf32>
    // CHECK:           %[[VAL_15:.*]] = IE.Reshape(%arg2, %[[VAL_4]]) : tensor<1x512xf32>, tensor<4xsi64> -> tensor<1x1x1x512xf32>
    // CHECK:           %[[VAL_16:.*]] = IE.FakeQuantize(%[[VAL_15]], %[[VAL_5]], %[[VAL_6]], %[[VAL_5]], %[[VAL_6]])
    // CHECK-SAME:          {auto_broadcast = "NUMPY", levels = 256 : i32} :
    // CHECK-SAME:          tensor<1x1x1x512xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>,
    // CHECK-SAME:          tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x512xf32>
    // CHECK:           %[[VAL_17:.*]] = IE.Reshape(%[[VAL_16]], %[[VAL_7]]) : tensor<1x1x1x512xf32>, tensor<2xsi64> -> tensor<1x512xf32>

    return %0, %2, %3 : tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>
    // CHECK:           return %[[VAL_10]], %[[VAL_14]], %[[VAL_17]] : tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>
}

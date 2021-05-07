// RUN: vpux-opt --canonicalize %s | FileCheck %s

func @main(%arg0: tensor<1x5x10x11xf16>) -> tensor<1x11x12x12xf16> {
    %0 = IE.Constant tensor<4xsi64> = dense<[0, 3, 0, 1]> : tensor<4xsi64>
    %1 = IE.Constant tensor<4xsi64> = dense<[0, 3, 2, 0]> : tensor<4xsi64>
    %2 = IE.Constant tensor<f16> = dense<1.000000e+00> : tensor<f16>
    // CHECK-NOT:   IE.Constant
    %3 = IE.Pad(%arg0, %0, %1, %2) {mode = "SYMMETRIC", operand_segment_sizes = dense<1> : vector<4xi32>} : tensor<1x5x10x11xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<f16> -> tensor<1x11x12x12xf16>
    // CHECK:       %[[VAL0:.*]] = IE.Pad(%arg0) {mode = "SYMMETRIC",
    // CHECK-SAME:  pad_value_attr = 1.000000e+00 : f32,
    // CHECK-SAME:  pads_begin_attr = [0 : i32, 3 : i32, 0 : i32, 1 : i32],
    // CHECK-SAME:  pads_end_attr = [0 : i32, 3 : i32, 2 : i32, 0 : i32]}
    // CHECK-SAME:  tensor<1x5x10x11xf16> -> tensor<1x11x12x12xf16>

    return %3 : tensor<1x11x12x12xf16>
    // CHECK:       return %[[VAL0]]
}

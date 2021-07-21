// RUN: vpux-opt --canonicalize %s | FileCheck %s

func @main(%arg0: tensor<1x3x10x10xf32>) -> tensor<1x3x20x15xf32> {
    %0 = const.Declare tensor<2xsi64> = #const.Content<dense<[20, 15]> : tensor<2xsi64>>
    %1 = const.Declare tensor<2xf32>  = #const.Content<dense<[2.000000e+00, 1.500000e+00]> : tensor<2xf32>>
    %2 = const.Declare tensor<2xsi64> = #const.Content<dense<[2, 3]> : tensor<2xsi64>>
    // CHECK-NOT:   const.Declare
    %3 = IE.Interpolate(%arg0, %0, %1, %2) {attr = {antialias = false, coord_mode = "half_pixel", cube_coeff = -7.500000e-01 : f32, mode = "nearest", nearest_mode = "round_prefer_floor", pads_begin = [0 : i32, 0 : i32, 0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32, 0 : i32, 0 : i32], shape_calc_mode = "sizes"}, operand_segment_sizes = dense<1> : vector<4xi32>} : tensor<1x3x10x10xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x3x20x15xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Interpolate(%arg0) {attr = {antialias = false, coord_mode = "half_pixel", cube_coeff = -7.500000e-01 : f32, mode = "nearest", nearest_mode = "round_prefer_floor", pads_begin = [0 : i32, 0 : i32, 0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32, 0 : i32, 0 : i32], shape_calc_mode = "sizes"},
    // CHECK-SAME: axes_attr = [2 : i32, 3 : i32],
    // CHECK-SAME: operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
    // CHECK-SAME: scales_attr = [2.000000e+00 : f32, 1.500000e+00 : f32],
    // CHECK-SAME: sizes_attr = [20 : i32, 15 : i32]}
    // CHECK-SAME: tensor<1x3x10x10xf32> -> tensor<1x3x20x15xf32>

    return %3 : tensor<1x3x20x15xf32>
    // CHECK:       return %[[VAL0]]
}

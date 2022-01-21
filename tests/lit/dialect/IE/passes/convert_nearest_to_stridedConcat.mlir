// RUN: vpux-opt --split-input-file --convert-nearest-to-strided-concat %s | FileCheck %s

// CHECK-LABEL: @ConvertNearestToStridedConcat_HW
func @ConvertNearestToStridedConcat_HW(%arg0: tensor<1x128x6x10xf32>) -> tensor<1x128x12x20xf32> {
    %0 = IE.Interpolate(%arg0) 
         {attr = {antialias = false, coord_mode = "asymmetric", cube_coeff = -7.500000e-01 : f64, 
         mode = "nearest", nearest_mode = "floor", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "sizes"}, 
         axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00],
         sizes_attr = [12, 20]} : 
         tensor<1x128x6x10xf32> -> tensor<1x128x12x20xf32>

    return %0 : tensor<1x128x12x20xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT1:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x10xf32>, tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>
    // CHECK: [[CONCAT2:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x10xf32>, tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>
    // CHECK: [[CONCAT_OUT:%.*]] = IE.Concat([[CONCAT1]], [[CONCAT2]]) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x20xf32>, tensor<1x128x6x20xf32> -> tensor<1x128x12x20xf32>
    // CHECK: return [[CONCAT_OUT]] : tensor<1x128x12x20xf32>
}

// CHECK-LABEL: @ConvertNearestToStridedConcat_H
func @ConvertNearestToStridedConcat_H(%arg0: tensor<1x128x6x10xf32>) -> tensor<1x128x12x10xf32> {
    %0 = IE.Interpolate(%arg0) 
         {attr = {antialias = false, coord_mode = "asymmetric", cube_coeff = -7.500000e-01 : f64, 
         mode = "nearest", nearest_mode = "floor", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "sizes"}, 
         axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 1.000000e+00],
         sizes_attr = [12, 10]} : 
         tensor<1x128x6x10xf32> -> tensor<1x128x12x10xf32>

    return %0 : tensor<1x128x12x10xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT_H:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x10xf32>, tensor<1x128x6x10xf32> -> tensor<1x128x12x10xf32>
    // CHECK: return [[CONCAT_H]] : tensor<1x128x12x10xf32>
}

// CHECK-LABEL: @ConvertNearestToStridedConcat_W
func @ConvertNearestToStridedConcat_W(%arg0: tensor<1x128x6x10xf32>) -> tensor<1x128x6x20xf32> {
    %0 = IE.Interpolate(%arg0) 
         {attr = {antialias = false, coord_mode = "asymmetric", cube_coeff = -7.500000e-01 : f64, 
         mode = "nearest", nearest_mode = "floor", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "sizes"}, 
         axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [1.000000e+00, 2.000000e+00],
         sizes_attr = [6, 20]} : 
         tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>

    return %0 : tensor<1x128x6x20xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT_W:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x10xf32>, tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>
    // CHECK: return [[CONCAT_W]] : tensor<1x128x6x20xf32>
}

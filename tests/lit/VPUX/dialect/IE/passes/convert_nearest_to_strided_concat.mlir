// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-nearest-to-strided-concat %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertNearestToStridedConcat_HW
func @ConvertNearestToStridedConcat_HW(%arg0: tensor<1x128x6x10xf32>) -> tensor<1x128x12x20xf32> {
    %0 = IE.Interpolate(%arg0) 
         {attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64, 
         mode = "NEAREST", nearest_mode = "FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, 
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
         {attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64, 
         mode = "NEAREST", nearest_mode = "FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, 
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
         {attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64, 
         mode = "NEAREST", nearest_mode = "FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, 
         axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [1.000000e+00, 2.000000e+00],
         sizes_attr = [6, 20]} : 
         tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>

    return %0 : tensor<1x128x6x20xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT_W:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x10xf32>, tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>
    // CHECK: return [[CONCAT_W]] : tensor<1x128x6x20xf32>
}

// CHECK-LABEL: @ConvertNearestToStridedConcatFQPropagation
func @ConvertNearestToStridedConcatFQPropagation(%arg0: tensor<1x128x6x10xf16>) -> tensor<1x128x12x20xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<0.000000e+00> : tensor<1x1x1x1xf16>>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<1.000000e+00> : tensor<1x1x1x1xf16>>

    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x128x6x10xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x6x10xf16>
  
    %1 = IE.Interpolate(%0) {attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64, mode = "NEAREST", nearest_mode = "FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [0.05328369140625, 0.0203399658203125], sizes_attr = [12, 20]} : tensor<1x128x6x10xf16> -> tensor<1x128x12x20xf16>
  
    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x128x12x20xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x12x20xf16>


    return %2 : tensor<1x128x12x20xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT_0:%.*]] = IE.Concat
    // CHECK-SAME: {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} 
    // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize
    // CHECK: [[CONCAT_1:%.*]] = IE.Concat
    // CHECK-SAME: {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} 
    // CHECK: [[FQ_1:%.*]] = IE.FakeQuantize
    // CHECK: IE.Concat([[FQ_0]], [[FQ_1]]) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x20xf16>, tensor<1x128x6x20xf16> -> tensor<1x128x12x20xf16>

}

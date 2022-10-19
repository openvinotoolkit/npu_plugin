// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-bilinear-to-strided-concat-and-conv --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertBilinearToStridedConcatAndConv_HW
func @ConvertBilinearToStridedConcatAndConv_HW(%arg0: tensor<1x32x96x176xf16>) -> tensor<1x32x192x352xf16> {
    %0 = IE.Interpolate(%arg0) 
         {attr = {antialias = false, coord_mode = "PYTORCH_HALF_PIXEL", cube_coeff = -7.500000e-01 : f64, mode = "LINEAR_ONNX", nearest_mode = "FLOOR", 
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SCALES"}, axes_attr = [2, 3], 
         operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 352]
         } : tensor<1x32x96x176xf16> -> tensor<1x32x192x352xf16>

    return %0 : tensor<1x32x192x352xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK: %cst = const.Declare tensor<32x1x2x2xf16> = #const.Content<dense<2.500000e-01> : tensor<32x1x2x2xf16>>
    // CHECK: [[CONCAT1:%.*]]  = IE.Concat(%arg0, %arg0) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x96x176xf16>, tensor<1x32x96x176xf16> -> tensor<1x32x96x352xf16>
    // CHECK: [[CONCAT2:%.*]]  = IE.Concat([[CONCAT1]], [[CONCAT1]]) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x96x352xf16>, tensor<1x32x96x352xf16> -> tensor<1x32x192x352xf16>
    // CHECK: [[SLICE1:%.*]] = IE.Slice [[CONCAT2]] [0, 0, 0, 351] [1, 32, 192, 1] : tensor<1x32x192x352xf16> to tensor<1x32x192x1xf16>
    // CHECK: [[CONCAT3:%.*]] = IE.Concat([[CONCAT2]], [[SLICE1]]) {
    // CHECK-SAME{LITERAL}:         static_offsets = [[0, 0, 0, 0], [0, 0, 0, 352]]
    // CHECK-SAME:              } : tensor<1x32x192x352xf16>, tensor<1x32x192x1xf16> -> tensor<1x32x192x353xf16>
    // CHECK: [[SLICE2:%.*]] = IE.Slice [[CONCAT3]] [0, 0, 191, 0] [1, 32, 1, 353] : tensor<1x32x192x353xf16> to tensor<1x32x1x353xf16>
    // CHECK: [[CONCAT4:%.*]] = IE.Concat([[CONCAT3]], [[SLICE2]]) {
    // CHECK-SAME{LITERAL}:         static_offsets = [[0, 0, 0, 0], [0, 0, 192, 0]]
    // CHECK-SAME:               } : tensor<1x32x192x353xf16>, tensor<1x32x1x353xf16> -> tensor<1x32x193x353xf16>
    // CHECK: [[RES:%.*]] = IE.GroupConvolution([[CONCAT4]], %cst) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x193x353xf16>, tensor<32x1x2x2xf16> -> tensor<1x32x192x352xf16>
    // CHECK: return [[RES]] : tensor<1x32x192x352xf16>
}

// CHECK-LABEL: @ConvertBilinearAlignCornersToStridedConcatAndConv_HW
func @ConvertBilinearAlignCornersToStridedConcatAndConv_HW(%arg0: tensor<1x32x96x176xf16>) -> tensor<1x32x191x351xf16> {
    %0 = IE.Interpolate(%arg0) {attr = {antialias = false, coord_mode = "ALIGN_CORNERS", cube_coeff = -7.500000e-01 : f64, mode = "LINEAR_ONNX", nearest_mode = "SIMPLE", 
        pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, axes_attr = [2, 3], 
        operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [1.9895833730697632, 1.9943181276321411], sizes_attr = [191, 351]
        } : tensor<1x32x96x176xf16> -> tensor<1x32x191x351xf16>

    return %0 : tensor<1x32x191x351xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK: %cst = const.Declare tensor<32x1x2x2xf16> = #const.Content<dense<2.500000e-01> : tensor<32x1x2x2xf16>>
    // CHECK: %0 = IE.Concat(%arg0, %arg0) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x96x176xf16>, tensor<1x32x96x176xf16> -> tensor<1x32x96x352xf16>
    // CHECK: %1 = IE.Concat(%0, %0) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x96x352xf16>, tensor<1x32x96x352xf16> -> tensor<1x32x192x352xf16>
    // CHECK: %2 = IE.GroupConvolution(%1, %cst) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x192x352xf16>, tensor<32x1x2x2xf16> -> tensor<1x32x191x351xf16> 
    // CHECK: return %2 : tensor<1x32x191x351xf16>
}

// CHECK-LABEL: @ConvertInterpolate
func @ConvertInterpolate(%arg0: tensor<1x256x1x1xf16>) -> tensor<1x256x32x32xf16> {
    %0 = IE.Interpolate(%arg0) 
         {attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64, mode = "LINEAR_ONNX", nearest_mode = "SIMPLE", 
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, axes_attr = [2, 3], 
         operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [32.000000e+00, 32.000000e+00], sizes_attr = [32, 32]
         } : tensor<1x256x1x1xf16> -> tensor<1x256x32x32xf16>

    return %0 : tensor<1x256x32x32xf16>

    // CHECK: %0 = IE.Interpolate(%arg0) {attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64, mode = "LINEAR_ONNX", nearest_mode = "SIMPLE", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [3.200000e+01, 3.200000e+01], sizes_attr = [32, 32]} : tensor<1x256x1x1xf16> -> tensor<1x256x32x32xf16>
    // CHECK: return %0 : tensor<1x256x32x32xf16>

}

// CHECK-LABEL: @NotConvertInterpolateWithChannelNeedAlign
func @NotConvertInterpolateWithChannelNeedAlign(%arg0: tensor<1x1x48x80xf16>) -> tensor<1x1x96x160xf16> {
    %0 = IE.Interpolate(%arg0) 
         {attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64, mode = "LINEAR_ONNX", nearest_mode = "SIMPLE", 
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, axes_attr = [2, 3], 
         operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]
         } : tensor<1x1x48x80xf16> -> tensor<1x1x96x160xf16>

    return %0 : tensor<1x1x96x160xf16>

    // CHECK: %0 = IE.Interpolate(%arg0) {attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64, mode = "LINEAR_ONNX", nearest_mode = "SIMPLE", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]} : tensor<1x1x48x80xf16> -> tensor<1x1x96x160xf16>
    // CHECK: return %0 : tensor<1x1x96x160xf16>

}

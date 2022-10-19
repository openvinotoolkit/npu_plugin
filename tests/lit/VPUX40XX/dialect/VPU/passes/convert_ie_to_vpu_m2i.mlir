// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX40XX" --convert-IE-to-VPU-M2I %s | FileCheck %s

// CHECK-LABEL: @convertColorConvert
func @convertColorConvert(%arg0: tensor<1x360x320x1xui8>) -> tensor<1x240x320x3xui8> {
   %0 = IE.YuvToRgb(%arg0) {inFmt = "NV12", operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, outFmt = "BGR"} : tensor<1x360x320x1xui8> -> tensor<1x240x320x3xui8>
   return %0 : tensor<1x240x320x3xui8>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.ColorConvert(%arg0) {inFmt = "NV12", outFmt = "BGR"} -> tensor<1x240x320x3xui8>
   //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @convertInterpolate
func @convertInterpolate(%arg0: tensor<1x3x256x256xui8>) -> tensor<1x3x224x224xui8> {
   %0 = IE.Interpolate(%arg0) {attr = {antialias = false, coord_mode = "ALIGN_CORNERS", cube_coeff = 0.000000e+00 : f64, mode = "NEAREST", nearest_mode = "ROUND_PREFER_FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [0.000000e+00, 0.000000e+00], sizes_attr = [224, 224]} : tensor<1x3x256x256xui8> -> tensor<1x3x224x224xui8>
   return %0 : tensor<1x3x224x224xui8>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.Resize(%arg0) {axes = [2, 3], sizes = [224, 224]} -> tensor<1x3x224x224xui8>
   //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @convertBatchNorm
func @convertBatchNorm(%arg0: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf16> {
   %0 = IE.BatchNormInference(%arg0) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], operand_segment_sizes = dense<[1, 0, 0, 0, 0]> : vector<5xi32>, variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16>
   return %0 : tensor<1x3x256x256xf16>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.Norm(%arg0) {beta_value = [0.000000e+00, 0.4169921875, 1.000000e+00], eps = 1.000000e-03 : f64, gamma_value = [0.000000e+00, 0.4169921875, 1.000000e+00], mean_value = [0.000000e+00, 0.4169921875, 1.000000e+00], variance_value = [7.826089859008789E-5, 1.3154296875, 7.5546875]} -> tensor<1x3x256x256xf16>
   //CHECK: return [[VAL0]]
}

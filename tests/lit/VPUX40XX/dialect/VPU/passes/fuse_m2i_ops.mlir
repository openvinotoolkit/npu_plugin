// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX40XX" --fuse-m2i-ops --canonicalize %s | FileCheck %s

// CHECK-LABEL: @fuseCscConvertResizePerm
func @fuseCscConvertResizePerm(%arg0: tensor<1x288x256x1xui8>) -> tensor<1x3x168x224xf16> {
   %0 = VPU.M2I.ColorConvert(%arg0) {inFmt = "NV12", outFmt = "RGB"} -> tensor<1x192x256x3xui8>
   %1 = VPU.Convert(%0) {dstElemType = f16} : tensor<1x192x256x3xui8> -> tensor<1x192x256x3xf16>
   %2 = VPU.Interpolate(%1) {attr = {antialias = false, coord_mode = "HALF_PIXEL", cube_coeff = -7.500000e-01 : f64, mode = "NEAREST", nearest_mode = "ROUND_PREFER_FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, axes_attr = [1, 2], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [168, 224]} : tensor<1x192x256x3xf16> -> tensor<1x168x224x3xf16>
   %3 = VPU.MemPermute(%2) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xf16> -> tensor<1x3x168x224xf16>
   return %3 : tensor<1x3x168x224xf16>

   //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], do_csc = true, do_norm = false, inFmt = "SP_NV12_8", outFmt = "PL_FP16_RGB", sizes = [168, 224]} -> tensor<1x3x168x224xf16>
   //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseCscResizePerm
func @fuseCscResizePerm(%arg0: tensor<1x288x256x1xui8>) -> tensor<1x3x168x224xui8> {
  %0 = VPU.M2I.ColorConvert(%arg0) {inFmt = "I420", outFmt = "BGR"} -> tensor<1x192x256x3xui8>
  %1 = VPU.M2I.Resize(%0) {axes = [1, 2], sizes = [168, 224]} -> tensor<1x168x224x3xui8>
  %2 = VPU.MemPermute(%1) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xui8> -> tensor<1x3x168x224xui8>
  return %2 : tensor<1x3x168x224xui8>

  //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], do_csc = true, do_norm = false, inFmt = "PL_YUV420_8", outFmt = "PL_BGR24", sizes = [168, 224]} -> tensor<1x3x168x224xui8>
  //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseCscResize
func @fuseCscResize(%arg0: tensor<1x288x256x1xui8>) -> tensor<1x168x224x3xui8> {
  %0 = VPU.M2I.ColorConvert(%arg0) {inFmt = "NV12", outFmt = "RGB"} -> tensor<1x192x256x3xui8>
  %1 = VPU.M2I.Resize(%0) {axes = [1, 2], sizes = [168, 224]} -> tensor<1x168x224x3xui8>
  return %1 : tensor<1x168x224x3xui8>

  //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {axes = [1, 2], do_csc = true, do_norm = false, inFmt = "SP_NV12_8", outFmt = "IL_RGB888", sizes = [168, 224]} -> tensor<1x168x224x3xui8>
  //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseCscPermute
func @fuseCscPermute(%arg0: tensor<1x252x224x1xui8>) -> tensor<1x3x168x224xui8> {
  %0 = VPU.M2I.ColorConvert(%arg0) {inFmt = "NV12", outFmt = "RGB"} -> tensor<1x168x224x3xui8>
  %1 = VPU.MemPermute(%0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xui8> -> tensor<1x3x168x224xui8>
  return %1 : tensor<1x3x168x224xui8>

  //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {do_csc = true, do_norm = false, inFmt = "SP_NV12_8", outFmt = "PL_RGB24"} -> tensor<1x3x168x224xui8>
  //CHECK: return [[VAL0]]
}

// -----

// CHECK-LABEL: @fuseCscConvertPerm
func @fuseCscConvertPerm(%arg0: tensor<1x252x224x1xui8>) -> tensor<1x3x168x224xf16> {
  %0 = VPU.M2I.ColorConvert(%arg0) {inFmt = "I420", outFmt = "RGB"} -> tensor<1x168x224x3xui8>
  %1 = VPU.Convert(%0) {dstElemType = f16} : tensor<1x168x224x3xui8> -> tensor<1x168x224x3xf16>
  %2 = VPU.MemPermute(%1) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x168x224x3xf16> -> tensor<1x3x168x224xf16>
  return %2 : tensor<1x3x168x224xf16>

  //CHECK: [[VAL0:%.*]] = VPU.M2I.Task(%arg0) {do_csc = true, do_norm = false, inFmt = "PL_YUV420_8", outFmt = "PL_FP16_RGB"} -> tensor<1x3x168x224xf16>
  //CHECK: return [[VAL0]]
}

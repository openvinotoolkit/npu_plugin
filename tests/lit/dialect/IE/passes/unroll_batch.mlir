// RUN: vpux-opt --split-input-file --unroll-batch %s | FileCheck %s

// CHECK-LABEL: @UnrollFullyConnectedBatch
func @UnrollFullyConnectedBatch(%arg0: tensor<2x16xf32>) -> tensor<2x64xf32> {
    %cst = const.Declare tensor<64x16xf16> = #const.Content<dense<1.0> : tensor<64x16xf32>, [#const.ConvertElemType<f16>]>
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<2x16xf32> -> tensor<2x16xf16>
    %1 = IE.FullyConnected(%0, %cst) : tensor<2x16xf16>, tensor<64x16xf16> -> tensor<2x64xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<2x64xf16> -> tensor<2x64xf32>

    return %2 : tensor<2x64xf32>

    // CHECK:       %[[WEIGHTS:.*]] = const.Declare tensor<64x16xf16> = #const.Content<dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x16xf32>, [#const.ConvertElemType<f16>]>
    // CHECK:       %[[INPUT:.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<2x16xf32> -> tensor<2x16xf16>
    // CHECK:       %[[INPUT_SLICE_1:.*]] = IE.Slice %[[INPUT]] [0, 0] [1, 16] : tensor<2x16xf16> to tensor<1x16xf16>
    // CHECK:       %[[INPUT_SLICE_2:.*]] = IE.Slice %[[INPUT]] [1, 0] [1, 16] : tensor<2x16xf16> to tensor<1x16xf16>
    // CHECK:       %[[FC_1:.*]] = IE.FullyConnected(%[[INPUT_SLICE_1]], %[[WEIGHTS]]) : tensor<1x16xf16>, tensor<64x16xf16> -> tensor<1x64xf16>
    // CHECK:       %[[FC_2:.*]] = IE.FullyConnected(%[[INPUT_SLICE_2]], %[[WEIGHTS]]) : tensor<1x16xf16>, tensor<64x16xf16> -> tensor<1x64xf16>
    // CHECK:       %[[FC_CONCAT:.*]] = IE.Concat(%[[FC_1]], %[[FC_2]])
    // CHECK-SAME:      {per_axis = {axis = 0 : i64}} : tensor<1x64xf16>, tensor<1x64xf16> -> tensor<2x64xf16>
    // CHECK:       %[[OUT:.*]] = IE.Convert(%[[FC_CONCAT]]) {dstElemType = f32} : tensor<2x64xf16> -> tensor<2x64xf32>
    // CHECK:       return %[[OUT]] : tensor<2x64xf32>
}

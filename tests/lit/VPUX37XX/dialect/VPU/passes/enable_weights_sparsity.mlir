// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --enable-weights-sparsity %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotSparsifyPaddedConv
func @DoNotSparsifyPaddedConv(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %arg1: tensor<16x1x1x4xsi32>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x3x1x1xf16>, [
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
    %1 = VPU.NCE.Convolution(%arg0, %weights, %arg1) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-NOT:   const.Sparsify
    // CHECK-NOT:   const.GetSparsityMap
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK:       [[weights:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:    : tensor<16x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
    // CHECK:       VPU.NCE.Convolution(%arg0, [[weights]], %arg1)
}

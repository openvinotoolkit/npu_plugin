// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --convert-IE-to-VPU-NCE %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToNCE
func @DepthConvToNCE(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> =
        #const.Content<dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]>

    %0 = IE.GroupConvolution(%arg0, %weights) {
            dilations = [1, 1],
            groups = 16,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = {attrs = {negative_slope = 0.1}, name = "IE.LeakyRelu"}
        } : tensor<1x16x40x80xf16, {order = #NHWC}>, tensor<16x1x4x8xf16, {order = #NHWC}>
            -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]])
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64,
    // CHECK-SAME:              lrelu_mult = 102 : i64, lrelu_shift = 10 : i64, mode = "LPRELU"},
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x37x73xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddWithReluRewriter
func @EltwiseAddWithReluRewriter(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = "NUMPY", post_op = {attrs = {}, name = "IE.ReLU"} } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1)
    // CHECK-SAME:      op_type = "ADD"
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = 0 : i64,
    // CHECK-SAME:              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "ADD", quant_mult = [16384], quant_post_shift = 0 : i64, quant_shift = [14]}}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB compilation-mode=DefaultHW" --convert-post-ops-to-ppe %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvEmptyPostOpRewriter
func @ConvEmptyPostOpRewriter(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> =
        #const.Content<dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>

    %0 = VPU.NCE.Convolution(%arg0, %cst0) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvWithReluRewriter
func @ConvWithReluRewriter(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> =
        #const.Content<dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>

    %0 = VPU.NCE.Convolution(%arg0, %cst0) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1],
            post_op = {attrs = {}, name = "IE.ReLU"}
        } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = 0 : i64,
    // CHECK-SAME:              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELU"},
    // CHECK-NOT:       post_op
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolWithClampRewriter
func @MaxPoolWithClampRewriter(%arg0: tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.MaxPool(%arg0) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1],
            kernel_size = [1, 1],
            post_op = {attrs = {max = 6.0, min = 0.0}, name = "IE.Clamp"}
        } : tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    return %0 : tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.MaxPool(%arg0) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-NOT:       post_op
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      ppe = {clamp_high = 393216 : i64, clamp_low = 0 : i64,
    // CHECK-SAME:              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}

    // CHECK:       return [[VAL0]] : tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvWithLReluRewriter
func @DepthConvWithLReluRewriter(%arg0: tensor<1x16x40x80xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %cst0 = const.Declare tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}> =
        #const.Content<dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]>

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst0) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1],
            post_op = {attrs = {negative_slope = 0.1}, name = "IE.LeakyRelu"}
        } : tensor<1x16x40x80xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>

    return %0 : tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.DepthConvolution(%arg0, [[CST]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-NOT:       post_op
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64,
    // CHECK-SAME:              lrelu_mult = 102 : i64, lrelu_shift = 10 : i64, mode = "LPRELU"},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddWithReluRewriter
func @EltwiseAddWithReluRewriter(%arg0: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { op_type = "ADD", post_op = {attrs = {}, name = "IE.ReLU"} } :
        tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = "ADD",
    // CHECK-NOT:       post_op
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = 0 : i64,
    // CHECK-SAME:              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "ADD"}}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseMultEmptyPostOpRewriter
func @EltwiseMultEmptyPostOpRewriter(%arg0: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { op_type = "MULTIPLY" } :
        tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = "MULTIPLY",
    // CHECK-SAME:      ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64,
    // CHECK-SAME:              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "MULT"}}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

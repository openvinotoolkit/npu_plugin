// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB compilation-mode=DefaultHW" --multi-cluster-strategy-assignment %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEClusterTilingSOH
func @ConvToNCEClusterTilingSOH(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]>
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>) 
    //CHECK-SAME    -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, strides = [1, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES0:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<80x64x3x3xf16, {order = #NHWC}>) 
    //CHECK-SAME    -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = MULTICASTED, num_tiles = [1, 1, 1, 1], kernel = [3, 3], pads = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, strides = [1, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES1:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<80x64x3x3xf16, {order = #NHWC}> -> tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<80x64x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>) 
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, strides = [1, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Convolution(%arg1, %arg2) {
    //CHECK-SAME:                 multiClusterStrategy = "SplitOverHeight", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, 
    //CHECK-SAME:                 rawFilterShape = [80, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = IE.Copy(%arg1) : tensor<1x80x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

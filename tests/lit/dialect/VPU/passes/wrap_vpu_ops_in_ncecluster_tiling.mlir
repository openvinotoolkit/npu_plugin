// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB compilation-mode=DefaultHW" --wrap-vpu-ops-in-ncecluster-tiling %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEClusterTilingSOK
func @ConvToNCEClusterTilingSOK(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = #const.Content<dense<10> : tensor<80x1x1x4xsi32>>
    %cst_0 = const.Declare tensor<80x128x1x1xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<80x128x1x1xf16>, [#const.Reorder<#NHWC>]>
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {multiClusterStrategy = "SplitOverKernel", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [80, 128, 1, 1], strides = [1, 1]} -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32> = #const.Content<dense<10> : tensor<64x1x1x4xsi32>>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x128x28x28xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 4 : i64}> {
    //CHECK:            [[RES0:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}> -> tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<64x128x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<64x128x1x1xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES1:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x128x1x1xf16, {order = #NHWC}> -> tensor<64x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTSTABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTSTABLE]] as %arg1: tensor<64x1x1x4xsi32>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = SEGMENTED, num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES2:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32> -> tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x128x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<64x128x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTSTABLE_CMX]] as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:   -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = DUPLICATED|SEGMENTED, num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES3:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
    //CHECK-SAME:                 pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:                 rawFilterShape = [64, 128, 1, 1], strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    //CHECK:            [[RES4:%.*]] = IE.Copy(%arg1) : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x28x28xf16, {order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

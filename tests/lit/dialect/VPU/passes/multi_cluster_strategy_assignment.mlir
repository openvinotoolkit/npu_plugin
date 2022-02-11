// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB compilation-mode=DefaultHW" --multi-cluster-strategy-assignment %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToMultiClusterSOH
func @ConvToMultiClusterSOH(%arg0: tensor<1x64x56x56xf16, {order = #NHWC}>) -> tensor<1x128x56x56xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<128x64x1x1xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<128x64x1x1xf16>, [#const.Reorder<#NHWC>]>
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [128, 64, 1, 1], strides = [1, 1]} -> tensor<1x128x56x56xf16, {order = #NHWC}>
    return %0 : tensor<1x128x56x56xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x16x16xf16, @CMX_NN, {kernel = [3, 3], mode = "overlapped", num_tiles = [1, 1, 4, 1], pads = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}> {
    //CHECK:            [[RES0:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}> -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>) -> !VPU.DistributedTensor<64x32x3x3xf16, @CMX_NN, {mode = "duplicated"}> {
    //CHECK:            [[RES1:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}> -> tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x64x16x16xf16, @CMX_NN, {mode = "segmented", num_tiles = [1, 1, 4, 1]}> {
    //CHECK:            [[RES2:%.*]] = VPU.NCE.Convolution(%arg1, %arg2) (bias : #const.Content<dense<1.000000e+00> : tensor<1x64x1x1xf16>>) {
    //CHECK-SAME:                 pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    //CHECK-SAME:                 strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    //CHECK:            [[RES3:%.*]] = IE.Copy(%arg1) {out_mem_space = @DDR} : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToMultiClusterSOK
func @ConvToMultiClusterSOH(%arg0: tensor<1x64x16x16xf16, {order = #NHWC}>) -> tensor<1x128x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<128x64x1x1xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<128x64x1x1xf16>, [#const.Reorder<#NHWC>]>
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [128, 64, 1, 1], strides = [1, 1]} -> tensor<1x128x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x128x16x16xf16, {order = #NHWC}>
}

// -----
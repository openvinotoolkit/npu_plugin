// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --multi-cluster-strategy-assignment %s | FileCheck %s
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvSubgraphOptimizationSplitOverKernel
func @ConvSubgraphOptimizationSplitOverKernel(%arg0: tensor<1x64x12x12xf16, {order = #NHWC}>) -> tensor<1x64x3x3xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<64x1x1x4xsi32> = #const.Content<dense<10> : tensor<64x1x1x4xsi32>>
    %cst_0 = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]>
    // supposed to be SOH without subgraph optimization
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [64, 64, 2, 2], strides = [2, 2]} -> tensor<1x64x6x6xf16, {order = #NHWC}>

    %cst1 = const.Declare tensor<64x1x1x4xsi32> = #const.Content<dense<10> : tensor<64x1x1x4xsi32>>
    %cst_1 = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]>
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst1) {pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, rawFilterShape = [64, 64, 2, 2], strides = [2, 2]} -> tensor<1x64x3x3xf16, {order = #NHWC}>
    return %1 : tensor<1x64x3x3xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE0:%.*]] = const.Declare tensor<64x1x1x4xsi32> = #const.Content<dense<10> : tensor<64x1x1x4xsi32>>
    //CHECK:        [[WEIGHTS0:%.*]] = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]>

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS0]], [[WEIGHTSTABLE0]])
    //CHECK-SAME:    multiClusterStrategy = "SplitOverKernel"

    //CHECK:        [[WEIGHTSTABLE1:%.*]] = const.Declare tensor<64x1x1x4xsi32> = #const.Content<dense<10> : tensor<64x1x1x4xsi32>>
    //CHECK:        [[WEIGHTS1:%.*]] = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]>
    //CHECK:        [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[WEIGHTS1]], [[WEIGHTSTABLE1]])
    //CHECK-SAME:    multiClusterStrategy = "SplitOverKernel"

    //CHECK:        return [[VAL1]] : tensor<1x64x3x3xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvSubgraphOptimizationClustering
func @ConvSubgraphOptimizationClustering(%arg0: tensor<1x80x22x22xf16, {order = #NHWC}>) -> tensor<1x48x22x22xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = #const.Content<dense<10> : tensor<80x1x1x4xsi32>>
    %cst_0 = const.Declare tensor<80x80x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<80x80x3x3xf16>, [#const.Reorder<#NHWC>]>
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [80, 80, 3, 3], strides = [1, 1]} -> tensor<1x80x22x22xf16, {order = #NHWC}>

    %cst1 = const.Declare tensor<64x1x1x4xsi32> = #const.Content<dense<10> : tensor<64x1x1x4xsi32>>
    %cst_1 = const.Declare tensor<64x80x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<64x80x3x3xf16>, [#const.Reorder<#NHWC>]>
    %1 = VPU.NCE.Convolution(%0, %cst_1, %cst1) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [64, 80, 3, 3], strides = [1, 1]} -> tensor<1x64x22x22xf16, {order = #NHWC}>

    %cst2 = const.Declare tensor<48x1x1x4xsi32> = #const.Content<dense<10> : tensor<48x1x1x4xsi32>>
    %cst_2 = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]>
    %2 = VPU.NCE.Convolution(%1, %cst_2, %cst2) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]} -> tensor<1x48x22x22xf16, {order = #NHWC}>

    return %2 : tensor<1x48x22x22xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE0:%.*]] = const.Declare tensor<80x1x1x4xsi32> = #const.Content<dense<10> : tensor<80x1x1x4xsi32>>
    //CHECK:        [[WEIGHTS0:%.*]] = const.Declare tensor<80x80x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<80x80x3x3xf16>, [#const.Reorder<#NHWC>]>

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS0]], [[WEIGHTSTABLE0]])
    //CHECK-SAME:    multiClusterStrategy = "SplitOverHeight"

    //CHECK:        [[WEIGHTSTABLE1:%.*]] = const.Declare tensor<64x1x1x4xsi32> = #const.Content<dense<10> : tensor<64x1x1x4xsi32>>
    //CHECK:        [[WEIGHTS1:%.*]] = const.Declare tensor<64x80x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<64x80x3x3xf16>, [#const.Reorder<#NHWC>]>
    //CHECK:        [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[WEIGHTS1]], [[WEIGHTSTABLE1]])
    //CHECK-SAME:    multiClusterStrategy = "SplitOverHeight"

    //CHECK:        [[WEIGHTSTABLE2:%.*]] = const.Declare tensor<48x1x1x4xsi32> = #const.Content<dense<10> : tensor<48x1x1x4xsi32>>
    //CHECK:        [[WEIGHTS2:%.*]] = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]>
    //CHECK:        [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], [[WEIGHTS2]], [[WEIGHTSTABLE2]])
    //CHECK-SAME:    multiClusterStrategy = "SplitOverHeight"

    //CHECK:        return [[VAL2]] : tensor<1x48x22x22xf16, {order = #NHWC}>
}

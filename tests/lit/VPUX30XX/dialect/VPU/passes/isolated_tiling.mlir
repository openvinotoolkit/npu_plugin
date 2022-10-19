// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --isolated-tiling --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @NoTilingConv
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<128x32x3x3xf16>,
// CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x128x1x1xf16>
func @NoTilingConv(
        %input: tensor<1x32x64x64xf16>,
        %filter: tensor<128x32x3x3xf16>,
        %bias: tensor<1x128x1x1xf16>)
            -> tensor<1x128x64x64xf16> {
    %1 = VPU.Convolution(%input, %filter, %bias) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16>, tensor<128x32x3x3xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: func @NoTilingMaxPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x125x125xf16>
func @NoTilingMaxPool(
        %input: tensor<1x16x125x125xf16>)
            -> tensor<1x16x125x125xf16> {
    %1 = VPU.MaxPool(%input) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x16x125x125xf16> -> tensor<1x16x125x125xf16>
    return %1 : tensor<1x16x125x125xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MaxPool([[INPUT]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = "FLOOR"
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x125x125xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x16x125x125xf16>
}

// -----

// CHECK-LABEL: func @NoTilingAdd
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1024x14x14xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x1024x14x14xf16>
func @NoTilingAdd(
        %input1: tensor<1x1024x14x14xf16>,
        %input2: tensor<1x1024x14x14xf16>)
            -> tensor<1x1024x14x14xf16> {
    %1 = VPU.Add(%input1, %input2) { auto_broadcast = "NUMPY" } : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16> -> tensor<1x1024x14x14xf16>
    return %1 : tensor<1x1024x14x14xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Add([[INPUT1]], [[INPUT2]])
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x1024x14x14xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NoTilingClusterNCEConv
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
func @NoTilingClusterNCEConv(%arg0: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<128x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = #const.Content<dense<10> : tensor<128x1x1x4xsi32, {mem_space = @CMX_NN}>>

    %0 = VPU.NCE.ClusterTiling (
            %arg0 as %arg1: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights as %arg2: tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights_table as %arg3: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
                -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
                rawFilterShape = [128, 32, 3, 3],
                strides = [1, 1]
            } -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
      VPU.Yield %1
    }

    return %0 : tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:        [[WEIGHT_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:        [[CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:          %arg0 as %arg1: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          [[WEIGHTS]] as %arg2: tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          [[WEIGHT_TABLE]] as %arg3: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:          -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           [[NCE_CONV:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:              pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield [[NCE_CONV]]

    // CHECK:         return [[CLUSTER_TILING]] : tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func @SplitNCEConvOverOC(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x128x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<128x1x1x4xsi32, {order = #NCHW}> = #const.Content<dense<10> : tensor<128x1x1x4xsi32>>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [128, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x128x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x128x64x64xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE_TILE1:%.+]] = const.Declare tensor<64x1x1x4xsi32> = #const.Content<dense<10>
    // CHECK-SAME:      : tensor<128x1x1x4xsi32>, [#const.SubView<[64, 0, 0, 0], [64, 1, 1, 4]>]>

    // CHECK:        [[FILTER_TILE1:%.+]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
    // CHECK-SAME:      : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[64, 0, 0, 0], [64, 32, 3, 3]>]>

    // CHECK:        [[WEIGHTS_TABLE_TILE0:%.+]] = const.Declare tensor<64x1x1x4xsi32> = #const.Content<dense<10>
    // CHECK-SAME:      : tensor<128x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 4]>]>

    // CHECK:        [[FILTER_TILE0:%.+]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
    // CHECK-SAME:      : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [64, 32, 3, 3]>]>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [64, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [64, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x64x64x64xf16, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 64, 0, 0]
    // CHECK-SAME:          -> tensor<1x128x64x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x128x64x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = type !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = type !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64x!qElemType0, {order = #NHWC}>
func @SplitQuantNCEConvOverOC(%arg0: tensor<1x32x64x64x!qElemType0, {order = #NHWC}>) -> tensor<1x256x64x64x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3x!qElemType2, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = #const.Content<dense<10> : tensor<256x1x1x4xsi32>>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x256x64x64x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x256x64x64x!qElemType1, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE_TILE1:%.+]] = const.Declare tensor<128x1x1x4xsi32> = #const.Content<dense<10>
    // CHECK-SAME:      : tensor<256x1x1x4xsi32>, [#const.SubView<[128, 0, 0, 0], [128, 1, 1, 4]>]>

    // CHECK:        [[FILTER_TILE1:%.+]] = const.Declare tensor<128x32x3x3x!qElemType2, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
    // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>, #const.SubView<[128, 0, 0, 0], [128, 32, 3, 3]>]>

    // CHECK:        [[WEIGHTS_TABLE_TILE0:%.+]] = const.Declare tensor<128x1x1x4xsi32> = #const.Content<dense<10>
    // CHECK-SAME:      : tensor<256x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [128, 1, 1, 4]>]>

    // CHECK:        [[FILTER_TILE0:%.+]] = const.Declare tensor<128x32x3x3x!qElemType2, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
    // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [128, 32, 3, 3]>]>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x128x64x64x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x128x64x64x!qElemType1, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 128, 0, 0]
    // CHECK-SAME:          -> tensor<1x256x64x64x!qElemType1, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEMaxPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x125x125xf16, {order = #NHWC}>)
func @SplitNCEMaxPoolOverH(%arg0: tensor<1x16x125x125xf16, {order = #NHWC}>) -> tensor<1x16x125x125xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = #const.Content<dense<10> : tensor<16x1x1x4xsi32>>
    %activation_window = const.Declare tensor<1x1x1x16xui8, {order = #NCHW}> = #const.Content<dense<1> : tensor<1x1x1x16xui8>>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 18 : i64,
        kernel_size = [3, 3],
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        strides = [1, 1]
    } -> tensor<1x16x125x125xf16, {order = #NHWC}>

    return %0 : tensor<1x16x125x125xf16, {order = #NHWC}>

    // CHECK:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NCHW}>
    // CHECK-SAME:      = #const.Content<dense<1> : tensor<1x1x1x16xui8>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}>
    // CHECK-SAME:      = #const.Content<dense<10> : tensor<16x1x1x4xsi32>>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 64, 125]
    // CHECK-SAME:      : tensor<1x16x125x125xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x64x125xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      activation_window_channel_length = 18 : i64,
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:      } -> tensor<1x16x63x125xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 62, 0] [1, 16, 63, 125]
    // CHECK-SAME:      : tensor<1x16x125x125xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x63x125xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      activation_window_channel_length = 18 : i64,
    // CHECK-SAME:      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:      } -> tensor<1x16x62x125xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 63, 0]
    // CHECK-SAME:      -> tensor<1x16x125x125xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x125x125xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @SplitNCEEltwiseAddOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x512x24x24xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x512x24x24xf16, {order = #NHWC}>
func @SplitNCEEltwiseAddOverC(
        %arg0: tensor<1x512x24x24xf16, {order = #NHWC}>,
        %arg1: tensor<1x512x24x24xf16, {order = #NHWC}>)
            -> tensor<1x512x24x24xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64,
               lrelu_shift = 0 : i64,
               mode = "ADD"}
    } -> tensor<1x512x24x24xf16, {order = #NHWC}>

    return %0 : tensor<1x512x24x24xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT0_TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 256, 24, 24]
    // CHECK-SAME:      : tensor<1x512x24x24xf16, {order = #NHWC}> to tensor<1x256x24x24xf16, {order = #NHWC}>

    // CHECK:       [[INPUT1_TILE0:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 0] [1, 256, 24, 24]
    // CHECK-SAME:      : tensor<1x512x24x24xf16, {order = #NHWC}> to tensor<1x256x24x24xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Eltwise([[INPUT0_TILE0]], [[INPUT1_TILE0]])
    // CHECK-SAME:      -> tensor<1x256x24x24xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT0_TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 256, 0, 0] [1, 256, 24, 24]
    // CHECK-SAME:      : tensor<1x512x24x24xf16, {order = #NHWC}> to tensor<1x256x24x24xf16, {order = #NHWC}>

    // CHECK:       [[INPUT1_TILE1:%.+]] = VPU.Slice [[INPUT2]] [0, 256, 0, 0] [1, 256, 24, 24]
    // CHECK-SAME:      : tensor<1x512x24x24xf16, {order = #NHWC}> to tensor<1x256x24x24xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Eltwise([[INPUT0_TILE1]], [[INPUT1_TILE1]])
    // CHECK-SAME:      -> tensor<1x256x24x24xf16, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 256, 0, 0]
    // CHECK-SAME:      -> tensor<1x512x24x24xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x512x24x24xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEEltwiseAddSameInput
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1024x14x14xf16, {order = #NHWC}>
func @SplitNCEEltwiseAddSameInput(%arg0: tensor<1x1024x14x14xf16, {order = #NHWC}>) -> tensor<1x1024x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = "ADD",
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64,
               lrelu_shift = 0 : i64,
               mode = "ADD"}
    } -> tensor<1x1024x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x1024x14x14xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 14, 14]
    // CHECK-SAME:      : tensor<1x1024x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x512x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Eltwise([[INPUT_TILE0]], [[INPUT_TILE0]]) {
    // CHECK-SAME:      op_type = "ADD"
    // CHECK-SAME:      } -> tensor<1x512x14x14xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 512, 0, 0] [1, 512, 14, 14]
    // CHECK-SAME:      : tensor<1x1024x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x512x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Eltwise([[INPUT_TILE1]], [[INPUT_TILE1]]) {
    // CHECK-SAME:      op_type = "ADD"
    // CHECK-SAME:      } -> tensor<1x512x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 512, 0, 0]
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x1024x14x14xf16, {order = #NHWC}>
}

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --isolated-tiling --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @SplitSwConvOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x32x3x3xf16>,
// CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x256x1x1xf16>
func @SplitSwConvOverOC(
        %input: tensor<1x32x64x64xf16>,
        %filter: tensor<256x32x3x3xf16>,
        %bias: tensor<1x256x1x1xf16>)
            -> tensor<1x256x64x64xf16> {
    %1 = VPU.Convolution(%input, %filter, %bias) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16>, tensor<256x32x3x3xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x64x64xf16>
    return %1 : tensor<1x256x64x64xf16>

    // Tile 0

    // CHECK:       [[FILTER_TILE0:%.+]] = VPU.Slice [[FILTER]] [0, 0, 0, 0] [128, 32, 3, 3]
    // CHECK-SAME:      : tensor<256x32x3x3xf16> to tensor<128x32x3x3xf16>

    // CHECK:       [[BIAS_TILE0:%.+]] = VPU.Slice [[BIAS]] [0, 0, 0, 0] [1, 128, 1, 1]
    // CHECK-SAME:      : tensor<1x256x1x1xf16> to tensor<1x128x1x1xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Convolution([[INPUT]], [[FILTER_TILE0]], [[BIAS_TILE0]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>

    // Tile 1

    // CHECK:       [[FILTER_TILE1:%.+]] = VPU.Slice [[FILTER]] [128, 0, 0, 0] [128, 32, 3, 3]
    // CHECK-SAME:      : tensor<256x32x3x3xf16> to tensor<128x32x3x3xf16>

    // CHECK:       [[BIAS_TILE1:%.+]] = VPU.Slice [[BIAS]] [0, 128, 0, 0] [1, 128, 1, 1]
    // CHECK-SAME:      : tensor<1x256x1x1xf16> to tensor<1x128x1x1xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Convolution([[INPUT]], [[FILTER_TILE1]], [[BIAS_TILE1]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 128, 0, 0]
    // CHECK-SAME:      -> tensor<1x256x64x64xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16>
}

// -----
// CHECK-LABEL: func @SplitSwMaxPoolOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16>
func @SplitSwMaxPoolOverH(
        %input: tensor<1x16x200x200xf16>)
            -> tensor<1x16x200x200xf16> {
    %1 = VPU.MaxPool(%input) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x16x200x200xf16> -> tensor<1x16x200x200xf16>
    return %1 : tensor<1x16x200x200xf16>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16> to tensor<1x16x101x200xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MaxPool([[INPUT_TILE0]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [0, 1]
    // CHECK-SAME:          rounding_type = "FLOOR"
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      -> tensor<1x16x100x200xf16>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 99, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16> to tensor<1x16x101x200xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MaxPool([[INPUT_TILE1]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [0, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = "FLOOR"
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      -> tensor<1x16x100x200xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 100, 0]
    // CHECK-SAME:      -> tensor<1x16x200x200xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x200x200xf16>
}

// -----

// CHECK-LABEL: func @SplitSwAddOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x2048x14x14xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x2048x14x14xf16>
func @SplitSwAddOverC(
        %input1: tensor<1x2048x14x14xf16>,
        %input2: tensor<1x2048x14x14xf16>)
            -> tensor<1x2048x14x14xf16> {
    %1 = VPU.Add(%input1, %input2) { auto_broadcast = "NUMPY" } : tensor<1x2048x14x14xf16>, tensor<1x2048x14x14xf16> -> tensor<1x2048x14x14xf16>
    return %1 : tensor<1x2048x14x14xf16>

    // Tile 0

    // CHECK:       [[INPUT0_TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[INPUT1_TILE0:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Add([[INPUT0_TILE0]], [[INPUT1_TILE0]])
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16>

    // Tile 1

    // CHECK:       [[INPUT0_TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 1024, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[INPUT1_TILE1:%.+]] = VPU.Slice [[INPUT2]] [0, 1024, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Add([[INPUT0_TILE1]], [[INPUT1_TILE1]])
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 1024, 0, 0]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x2048x14x14xf16>
}

// -----

// CHECK-LABEL: func @SplitAddSameInputOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x2048x14x14xf16>
func @SplitAddSameInputOverC(
        %input: tensor<1x2048x14x14xf16>)
            -> tensor<1x2048x14x14xf16> {
    %1 = VPU.And(%input, %input) { auto_broadcast = "NUMPY" } : tensor<1x2048x14x14xf16>, tensor<1x2048x14x14xf16> -> tensor<1x2048x14x14xf16>
    return %1 : tensor<1x2048x14x14xf16>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:       : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.And([[INPUT_TILE0]], [[INPUT_TILE0]])
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 1024, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.And([[INPUT_TILE1]], [[INPUT_TILE1]])
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 1024, 0, 0]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x2048x14x14xf16>
}

// -----

func @InterpSplitOverH(
        %input1: tensor<1x32x64x64xf16>)
            -> tensor<1x32x256x256xf16> {

    %0 = const.Declare tensor<2xsi64> = #const.Content<dense<[256, 256]> : tensor<2xsi64>>
    %1 = const.Declare tensor<2xf32>  = #const.Content<dense<[4.000000e+00, 4.00000e+00]> : tensor<2xf32>>
    %2 = const.Declare tensor<2xsi64> = #const.Content<dense<[2, 3]> : tensor<2xsi64>>

    %3 = VPU.Interpolate(%input1, %0, %1, %2) {
            attr = {antialias = false, coord_mode = "HALF_PIXEL", cube_coeff = -7.500000e-01, mode = "LINEAR", nearest_mode = "ROUND_PREFER_FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"},
            operand_segment_sizes = dense<1> : vector<4xi32> } :
        tensor<1x32x64x64xf16>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x32x256x256xf16>

    return %3 : tensor<1x32x256x256xf16>
}

// CHECK-LABEL: func @InterpSplitOverH
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x32x64x64xf16>

// Tile 0

// CHECK:       [[TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 32, 21, 64]
// CHECK-SAME:      : tensor<1x32x64x64xf16> to tensor<1x32x21x64xf16>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 2, 0]
// CHECK-SAME:      : tensor<1x32x21x64xf16>
// CHECK-SAME:      -> tensor<1x32x86x256xf16>

// Tile 1

// CHECK:       [[TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 21, 0] [1, 32, 21, 64]
// CHECK-SAME:      : tensor<1x32x64x64xf16> to tensor<1x32x21x64xf16>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1, 0]
// CHECK-SAME:      : tensor<1x32x21x64xf16>
// CHECK-SAME:      -> tensor<1x32x85x256xf16>

// Tile 2

// CHECK:       [[TILE2:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 42, 0] [1, 32, 21, 64]
// CHECK-SAME:      : tensor<1x32x64x64xf16> to tensor<1x32x21x64xf16>
// CHECK:       [[INTERP2:%.+]] = VPU.Interpolate([[TILE2]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1, 0]
// CHECK-SAME:      : tensor<1x32x21x64xf16>
// CHECK-SAME:      -> tensor<1x32x85x256xf16>

// Concat

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]], [[INTERP2]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 171, 0]
// CHECK-SAME:      -> tensor<1x32x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x32x256x256xf16>

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
func @SplitNCEConvOverOC(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = #const.Content<dense<10> : tensor<256x1x1x4xsi32>>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE_TILE1:%.+]] = const.Declare tensor<128x1x1x4xsi32> = #const.Content<dense<10>
    // CHECK-SAME:      : tensor<256x1x1x4xsi32>, [#const.SubView<[128, 0, 0, 0], [128, 1, 1, 4]>]>

    // CHECK:        [[FILTER_TILE1:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
    // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[128, 0, 0, 0], [128, 32, 3, 3]>]>

    // CHECK:        [[WEIGHTS_TABLE_TILE0:%.+]] = const.Declare tensor<128x1x1x4xsi32> = #const.Content<dense<10>
    // CHECK-SAME:      : tensor<256x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [128, 1, 1, 4]>]>

    // CHECK:        [[FILTER_TILE0:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
    // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [128, 32, 3, 3]>]>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x128x64x64xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x128x64x64xf16, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 128, 0, 0]
    // CHECK-SAME:          -> tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = type !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = type !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64x!qElemType0, {order = #NHWC}>
func @SplitQuantNCEConvOverOC(%arg0: tensor<1x32x64x64x!qElemType0, {order = #NHWC}>) -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<512x32x3x3x!qElemType2, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<512x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<512x1x1x4xsi32, {order = #NCHW}> = #const.Content<dense<10> : tensor<512x1x1x4xsi32>>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [512, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE_TILE1:%.+]] = const.Declare tensor<256x1x1x4xsi32> = #const.Content<dense<10>
    // CHECK-SAME:      : tensor<512x1x1x4xsi32>, [#const.SubView<[256, 0, 0, 0], [256, 1, 1, 4]>]>

    // CHECK:        [[FILTER_TILE1:%.+]] = const.Declare tensor<256x32x3x3x!qElemType2, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
    // CHECK-SAME:      : tensor<512x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>, #const.SubView<[256, 0, 0, 0], [256, 32, 3, 3]>]>

    // CHECK:        [[WEIGHTS_TABLE_TILE0:%.+]] = const.Declare tensor<256x1x1x4xsi32> = #const.Content<dense<10>
    // CHECK-SAME:      : tensor<512x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [256, 1, 1, 4]>]>

    // CHECK:        [[FILTER_TILE0:%.+]] = const.Declare tensor<256x32x3x3x!qElemType2, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
    // CHECK-SAME:      : tensor<512x32x3x3xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [256, 32, 3, 3]>]>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x256x64x64x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x256x64x64x!qElemType1, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 256, 0, 0]
    // CHECK-SAME:          -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEMaxPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16, {order = #NHWC}>)
func @SplitNCEMaxPoolOverH(%arg0: tensor<1x16x200x200xf16, {order = #NHWC}>) -> tensor<1x16x200x200xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = #const.Content<dense<10> : tensor<16x1x1x4xsi32>>
    %activation_window = const.Declare tensor<1x1x1x16xui8, {order = #NCHW}> = #const.Content<dense<1> : tensor<1x1x1x16xui8>>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 18 : i64,
        kernel_size = [3, 3],
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        strides = [1, 1]
    } -> tensor<1x16x200x200xf16, {order = #NHWC}>

    return %0 : tensor<1x16x200x200xf16, {order = #NHWC}>

    // CHECK:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NCHW}>
    // CHECK-SAME:      = #const.Content<dense<1> : tensor<1x1x1x16xui8>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}>
    // CHECK-SAME:      = #const.Content<dense<10> : tensor<16x1x1x4xsi32>>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x101x200xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      activation_window_channel_length = 18 : i64,
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:      } -> tensor<1x16x100x200xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 99, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x101x200xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      activation_window_channel_length = 18 : i64,
    // CHECK-SAME:      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:      } -> tensor<1x16x100x200xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 100, 0]
    // CHECK-SAME:      -> tensor<1x16x200x200xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x200x200xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @SplitNCEEltwiseAddOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1024x24x24xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x1024x24x24xf16, {order = #NHWC}>
func @SplitNCEEltwiseAddOverC(
        %arg0: tensor<1x1024x24x24xf16, {order = #NHWC}>,
        %arg1: tensor<1x1024x24x24xf16, {order = #NHWC}>)
            -> tensor<1x1024x24x24xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64,
               lrelu_shift = 0 : i64,
               mode = "ADD"}
    } -> tensor<1x1024x24x24xf16, {order = #NHWC}>

    return %0 : tensor<1x1024x24x24xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT0_TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 512, 24, 24]
    // CHECK-SAME:      : tensor<1x1024x24x24xf16, {order = #NHWC}> to tensor<1x512x24x24xf16, {order = #NHWC}>

    // CHECK:       [[INPUT1_TILE0:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 0] [1, 512, 24, 24]
    // CHECK-SAME:      : tensor<1x1024x24x24xf16, {order = #NHWC}> to tensor<1x512x24x24xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Eltwise([[INPUT0_TILE0]], [[INPUT1_TILE0]])
    // CHECK-SAME:      -> tensor<1x512x24x24xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT0_TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 512, 0, 0] [1, 512, 24, 24]
    // CHECK-SAME:      : tensor<1x1024x24x24xf16, {order = #NHWC}> to tensor<1x512x24x24xf16, {order = #NHWC}>

    // CHECK:       [[INPUT1_TILE1:%.+]] = VPU.Slice [[INPUT2]] [0, 512, 0, 0] [1, 512, 24, 24]
    // CHECK-SAME:      : tensor<1x1024x24x24xf16, {order = #NHWC}> to tensor<1x512x24x24xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Eltwise([[INPUT0_TILE1]], [[INPUT1_TILE1]])
    // CHECK-SAME:      -> tensor<1x512x24x24xf16, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 512, 0, 0]
    // CHECK-SAME:      -> tensor<1x1024x24x24xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x1024x24x24xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEEltwiseAddSameInput
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2048x14x14xf16, {order = #NHWC}>
func @SplitNCEEltwiseAddSameInput(%arg0: tensor<1x2048x14x14xf16, {order = #NHWC}>) -> tensor<1x2048x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = "ADD",
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64,
               lrelu_shift = 0 : i64,
               mode = "ADD"}
    } -> tensor<1x2048x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x2048x14x14xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Eltwise([[INPUT_TILE0]], [[INPUT_TILE0]]) {
    // CHECK-SAME:      op_type = "ADD"
    // CHECK-SAME:      } -> tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 1024, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Eltwise([[INPUT_TILE1]], [[INPUT_TILE1]]) {
    // CHECK-SAME:      op_type = "ADD"
    // CHECK-SAME:      } -> tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 1024, 0, 0]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x2048x14x14xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @ConvertU8F32SplitOverH(%arg0: tensor<1x2x80x4000xui8, {order = #NHWC}>) -> tensor<1x2x80x4000xf32, {order = #NHWC}> {
  %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x2x80x4000xui8, {order = #NHWC}> -> tensor<1x2x80x4000xf32, {order = #NHWC}>
  return %0 : tensor<1x2x80x4000xf32, {order = #NHWC}>
}

// CHECK-LABEL: @ConvertU8F32SplitOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2x80x4000xui8, {order = #NHWC}>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2, 40, 4000]
// CHECK-SAME:      : tensor<1x2x80x4000xui8, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x2x40x4000xui8, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Convert([[INPUT_TILE0]]) {
// CHECK-SAME:      dstElemType = f32
// CHECK-SAME:      }> -> tensor<1x2x40x4000xf32, {order = #NHWC}>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 2, 40, 4000]
// CHECK-SAME:      : tensor<1x2x80x4000xui8, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x2x40x4000xui8, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Convert([[INPUT_TILE1]]) {
// CHECK-SAME:      dstElemType = f32
// CHECK-SAME:      }> -> tensor<1x2x40x4000xf32, {order = #NHWC}>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:      -> tensor<1x2x80x4000xf32, {order = #NHWC}>

// CHECK:       return [[OUTPUT]] : tensor<1x2x80x4000xf32, {order = #NHWC}>

// -----

func @SigmoidSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Sigmoid(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SigmoidSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sigmoid([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sigmoid([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @TanhSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Tanh(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @TanhSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tanh([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tanh([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @ExpSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Exp(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @ExpSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Exp([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Exp([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @SqrtSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Sqrt(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SqrtSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sqrt([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sqrt([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @EluSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Elu(%arg0) {x = 1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @EluSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Elu([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1], x = 1.000000e+00 : f64} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Elu([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1], x = 1.000000e+00 : f64} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @HSwishSplitOverH(%arg0: tensor<1x16x80x1280xf16>) -> tensor<1x16x80x1280xf16> {
  %0 = VPU.HSwish(%arg0) : tensor<1x16x80x1280xf16> -> tensor<1x16x80x1280xf16>
  return %0 : tensor<1x16x80x1280xf16>
}

// CHECK-LABEL: @HSwishSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x16x80x1280xf16>) -> tensor<1x16x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 20, 1280]
// CHECK-SAME:  : tensor<1x16x80x1280xf16> to tensor<1x16x20x1280xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.HSwish([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 4, 1]} : tensor<1x16x20x1280xf16> -> tensor<1x16x20x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 20, 0] [1, 16, 20, 1280]
// CHECK-SAME:  : tensor<1x16x80x1280xf16> to tensor<1x16x20x1280xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.HSwish([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 4, 1]} : tensor<1x16x20x1280xf16> -> tensor<1x16x20x1280xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 16, 20, 1280]
// CHECK-SAME:  : tensor<1x16x80x1280xf16> to tensor<1x16x20x1280xf16>
// CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.HSwish([[INPUT_TILE2]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 4, 1]} : tensor<1x16x20x1280xf16> -> tensor<1x16x20x1280xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 60, 0] [1, 16, 20, 1280]
// CHECK-SAME:  : tensor<1x16x80x1280xf16> to tensor<1x16x20x1280xf16>
// CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.HSwish([[INPUT_TILE3]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 4, 1]} : tensor<1x16x20x1280xf16> -> tensor<1x16x20x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 40, 0], [0, 0, 60, 0]
// CHECK-SAME:  : tensor<1x16x20x1280xf16>, tensor<1x16x20x1280xf16>, tensor<1x16x20x1280xf16>,
// CHECK-SAME:   tensor<1x16x20x1280xf16> -> tensor<1x16x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x16x80x1280xf16>

// -----

func @SplitDivideEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Divide(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitDivideEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Divide([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Divide([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func @MemPermuteSplitNCHWToNHWC2Part(%arg0: tensor<1x546x40x40xf16>) -> tensor<1x40x40x546xf16> {
  %0 = VPU.MemPermute(%arg0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x546x40x40xf16> -> tensor<1x40x40x546xf16>
  return %0 : tensor<1x40x40x546xf16>
}
// CHECK-LABEL: @MemPermuteSplitNCHWToNHWC2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x546x40x40xf16>) -> tensor<1x40x40x546xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 546, 20, 40]
// CHECK-SAME:  : tensor<1x546x40x40xf16> to tensor<1x546x20x40xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MemPermute([[INPUT_TILE0]]) {
// CHECK-SAME:  dst_order = #NCHW, mem_perm = #NHWC, tilingStrategy = [1, 2, 1, 1]
// CHECK-SAME:  } : tensor<1x546x20x40xf16> -> tensor<1x20x40x546xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 20, 0] [1, 546, 20, 40]
// CHECK-SAME:  : tensor<1x546x40x40xf16> to tensor<1x546x20x40xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MemPermute([[INPUT_TILE1]]) {
// CHECK-SAME:  dst_order = #NCHW, mem_perm = #NHWC, tilingStrategy = [1, 2, 1, 1]
// CHECK-SAME:  } : tensor<1x546x20x40xf16> -> tensor<1x20x40x546xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 20, 0, 0]
// CHECK-SAME:  : tensor<1x20x40x546xf16>, tensor<1x20x40x546xf16> -> tensor<1x40x40x546xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x40x40x546xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func @AvgPoolSwSplit2Part(%arg0: tensor<1x32x1800x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>) -> tensor<1x32x1789x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
  %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x32x1800x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x32x1789x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  return %0 : tensor<1x32x1789x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
}
// CHECK-LABEL: @AvgPoolSwSplit2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x1800x16xf16, {order = #NHWC}>) -> tensor<1x32x1789x16xf16, {order = #NHWC}> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 906, 16]
// CHECK-SAME:  :  tensor<1x32x1800x16xf16, {order = #NHWC}> to tensor<1x32x906x16xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.AvgPool([[INPUT_TILE0]]) {
// CHECK-SAME:  exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
// CHECK-SAME:  } : tensor<1x32x906x16xf16, {order = #NHWC}> -> tensor<1x32x895x16xf16, {order = #NHWC}>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 895, 0] [1, 32, 905, 16]
// CHECK-SAME:  : tensor<1x32x1800x16xf16, {order = #NHWC}> to tensor<1x32x905x16xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.AvgPool([[INPUT_TILE1]]) {
// CHECK-SAME:  exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
// CHECK-SAME:  } : tensor<1x32x905x16xf16, {order = #NHWC}> -> tensor<1x32x894x16xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 895, 0]
// CHECK-SAME:  : tensor<1x32x895x16xf16, {order = #NHWC}>, tensor<1x32x894x16xf16, {order = #NHWC}> -> tensor<1x32x1789x16xf16, {order = #NHWC}>

// CHECK:       return [[OUTPUT]] : tensor<1x32x1789x16xf16, {order = #NHWC}>

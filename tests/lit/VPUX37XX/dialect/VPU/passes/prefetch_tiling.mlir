// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --prefetch-tiling --canonicalize %s | FileCheck %s

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
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
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
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 128, 0, 0]
    // CHECK-SAME:      -> tensor<1x256x64x64xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16>
}

// -----

// CHECK-LABEL: func @SplitSwMaxPoolOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x244x244xf16>
func @SplitSwMaxPoolOverH(
        %input: tensor<1x16x244x244xf16>)
            -> tensor<1x16x244x244xf16> {
    %1 = VPU.MaxPool(%input) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x16x244x244xf16> -> tensor<1x16x244x244xf16>
    return %1 : tensor<1x16x244x244xf16>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 123, 244]
    // CHECK-SAME:       : tensor<1x16x244x244xf16> to tensor<1x16x123x244xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MaxPool([[INPUT_TILE0]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [0, 1]
    // CHECK-SAME:          rounding_type = "FLOOR"
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      -> tensor<1x16x122x244xf16>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 121, 0] [1, 16, 123, 244]
    // CHECK-SAME:      : tensor<1x16x244x244xf16> to tensor<1x16x123x244xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MaxPool([[INPUT_TILE1]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [0, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = "FLOOR"
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      -> tensor<1x16x122x244xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 122, 0]
    // CHECK-SAME:      -> tensor<1x16x244x244xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x244x244xf16>
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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @GenericTiling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x144x20x20xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<144x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<576x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE1:%arg[0-9]]]: tensor<144x1x1x4xsi32, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE2:%arg[0-9]]]: tensor<576x1x1x4xsi32, {order = #NHWC}>
func @GenericTiling(
        %input: tensor<1x144x20x20xf16, {order = #NHWC}>,
        %weights1: tensor<144x144x3x3xf16, {order = #NHWC}>,
        %weights2: tensor<576x144x3x3xf16, {order = #NHWC}>,
        %weights_table1: tensor<144x1x1x4xsi32, {order = #NHWC}>,
        %weights_table2: tensor<576x1x1x4xsi32, {order = #NHWC}>)
            -> tensor<1x576x20x20xf16, {order = #NHWC}> {
    %1 = VPU.NCE.Convolution(%input, %weights1, %weights_table1) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [144, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<144x144x3x3xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%1, %1) {op_type = "ADD"} : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<1x144x20x20xf16, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %weights2, %weights_table2) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [576, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<576x144x3x3xf16, {order = #NHWC}>, tensor<576x1x1x4xsi32, {order = #NHWC}> -> tensor<1x576x20x20xf16, {order = #NHWC}>
    return %3 : tensor<1x576x20x20xf16, {order = #NHWC}>

    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:     {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [144, 144, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // CHECK:       [[AND:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[CONV_1]]) {op_type = "ADD"}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[WEIGHTS_TILE0:%.+]] = VPU.Slice [[WEIGHTS2]] [0, 0, 0, 0] [288, 144, 3, 3]
    // CHECK-SAME:      tensor<576x144x3x3xf16, {order = #NHWC}> to tensor<288x144x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE_TILE0:%.+]] = VPU.Slice [[WEIGHTS_TABLE2]] [0, 0, 0, 0] [288, 1, 1, 4]
    // CHECK-SAME:      tensor<576x1x1x4xsi32, {order = #NHWC}> to tensor<288x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:     {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [288, 144, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:          -> tensor<1x288x20x20xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[WEIGHTS_TILE1:%.+]] = VPU.Slice [[WEIGHTS2]] [288, 0, 0, 0] [288, 144, 3, 3]
    // CHECK-SAME:      tensor<576x144x3x3xf16, {order = #NHWC}> to tensor<288x144x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE_TILE1:%.+]] = VPU.Slice [[WEIGHTS_TABLE2]] [288, 0, 0, 0] [288, 1, 1, 4]
    // CHECK-SAME:      tensor<576x1x1x4xsi32, {order = #NHWC}> to tensor<288x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:     {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [288, 144, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:          -> tensor<1x288x20x20xf16, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 288, 0, 0]
    // CHECK-SAME:      -> tensor<1x576x20x20xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x576x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func @SplitNCEConvOverOC(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = #const.Content<dense<1> : tensor<256x1x1x4xsi32>>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE_TILE1:%.+]] = const.Declare tensor<128x1x1x4xsi32> = #const.Content<dense<1>
    // CHECK-SAME:      : tensor<256x1x1x4xsi32>, [#const.SubView<[128, 0, 0, 0], [128, 1, 1, 4]>]>

    // CHECK:        [[FILTER_TILE1:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
    // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[128, 0, 0, 0], [128, 32, 3, 3]>]>

    // CHECK:        [[WEIGHTS_TABLE_TILE0:%.+]] = const.Declare tensor<128x1x1x4xsi32> = #const.Content<dense<1>
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

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 128, 0, 0]
    // CHECK-SAME:          -> tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x340x340xf16, {order = #NHWC}>)
func @SplitNCEPoolOverH(%arg0: tensor<1x16x340x340xf16, {order = #NHWC}>) -> tensor<1x16x340x340xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>
    %activation_window = const.Declare tensor<1x1x1x16xui8> = #const.Content<dense<1> : tensor<1x1x1x16xui8>>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 18 : i64,
        kernel_size = [3, 3],
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        strides = [1, 1]
    } -> tensor<1x16x340x340xf16, {order = #NHWC}>

    return %0 : tensor<1x16x340x340xf16, {order = #NHWC}>

    // CHECK:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>
    // CHECK-SAME:      = #const.Content<dense<1> : tensor<1x1x1x16xui8>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:      = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 58, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x58x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:      } -> tensor<1x16x57x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 56, 0] [1, 16, 59, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x59x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x57x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 113, 0] [1, 16, 59, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x59x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE2]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x57x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 170, 0] [1, 16, 59, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x59x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE3]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x57x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 227, 0] [1, 16, 58, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x58x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE4]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x56x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 283, 0] [1, 16, 57, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x57x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE5]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x56x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]]) {
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 57, 0], [0, 0, 114, 0], [0, 0, 171, 0], [0, 0, 228, 0], [0, 0, 284, 0]
    // CHECK-SAME:      -> tensor<1x16x340x340xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x340x340xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoTileWithSOH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {order = #NHWC}>
func @NoTileWithSOH(
        %arg0: tensor<1x32x100x100xf16, {order = #NHWC}>)
            -> tensor<1x128x100x100xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
        : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<128x1x1x4xsi32> = #const.Content<dense<1>
        : tensor<128x1x1x4xsi32>>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverHeight",
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [128, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x128x100x100xf16, {order = #NHWC}>

    return %0 : tensor<1x128x100x100xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32>
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}>
    // CHECK-NOT:   VPU.Slice

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x128x100x100xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x128x100x100xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TileWithSOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x210x210xf16, {order = #NHWC}>
func @TileWithSOH(
        %arg0: tensor<1x16x210x210xf16, {order = #NHWC}>)
            -> tensor<1x32x210x210xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
        : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = #const.Content<dense<1>
        : tensor<32x1x1x4xsi32>>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverHeight",
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [32, 16, 3, 3],
        strides = [1, 1]
    } -> tensor<1x32x210x210xf16, {order = #NHWC}>

    return %0 : tensor<1x32x210x210xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 106, 210]
    // CHECK-SAME:          tensor<1x16x210x210xf16, {order = #NHWC}> to tensor<1x16x106x210xf16, {order = #NHWC}>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
    // CHECK-SAME:          tensor<1x32x105x210xf16, {order = #NHWC}>

    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 104, 0] [1, 16, 106, 210]
    // CHECK-SAME:          tensor<1x16x210x210xf16, {order = #NHWC}> to tensor<1x16x106x210xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x32x105x210xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV1]], [[CONV2]])

    // CHECK:       return [[CONCAT]] : tensor<1x32x210x210xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoTileWithSOK
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x10x10xf16, {order = #NHWC}>
func @NoTileWithSOK(
        %arg0: tensor<1x32x10x10xf16, {order = #NHWC}>)
            -> tensor<1x240x10x10xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
        : tensor<240x32x7x7xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<240x1x1x4xsi32> = #const.Content<dense<1>
        : tensor<240x1x1x4xsi32>>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverKernel",
        pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64},
        rawFilterShape = [240, 32, 7, 7],
        strides = [1, 1]
    } -> tensor<1x240x10x10xf16, {order = #NHWC}>

    return %0 : tensor<1x240x10x10xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<240x1x1x4xsi32>
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}>
    // CHECK-NOT:   VPU.Slice

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64},
    // CHECK-SAME:          rawFilterShape = [240, 32, 7, 7],
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x240x10x10xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x240x10x10xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TileWithSOK
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x30x30xf16, {order = #NHWC}>
func @TileWithSOK(
        %arg0: tensor<1x32x30x30xf16, {order = #NHWC}>)
            -> tensor<1x768x30x30xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
        : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<768x1x1x4xsi32> = #const.Content<dense<1>
        : tensor<768x1x1x4xsi32>>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverKernel",
        pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64},
        rawFilterShape = [768, 32, 7, 7],
        strides = [1, 1]
    } -> tensor<1x768x30x30xf16, {order = #NHWC}>

    return %0 : tensor<1x768x30x30xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<384x1x1x4xsi32>
    // CHECK-SAME:          #const.SubView<[384, 0, 0, 0], [384, 1, 1, 4]>]>
    // CHECK:       [[WEIGHTS2:%.+]] = const.Declare tensor<384x32x7x7xf16, {order = #NHWC}>
    // CHECK-SAME:          #const.SubView<[384, 0, 0, 0], [384, 32, 7, 7]>
    // CHECK:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<384x1x1x4xsi32>
    // CHECK-SAME:          #const.SubView<[0, 0, 0, 0], [384, 1, 1, 4]>
    // CHECK:       [[WEIGHTS1:%.+]] = const.Declare tensor<384x32x7x7xf16, {order = #NHWC}>
    // CHECK-SAME:          #const.SubView<[0, 0, 0, 0], [384, 32, 7, 7]>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64}
    // CHECK-SAME:          rawFilterShape = [384, 32, 7, 7]
    // CHECK-SAME:          tensor<1x384x30x30xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64}
    // CHECK-SAME:          rawFilterShape = [384, 32, 7, 7]
    // CHECK-SAME:          tensor<1x384x30x30xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV1]], [[CONV2]])

    // CHECK:       return [[CONCAT]] : tensor<1x768x30x30xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @LargeConstPipeliningSOKFor
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x256x14x14xf16, {order = #NHWC}>
func @LargeConstPipeliningSOKFor(
        %arg0: tensor<1x256x14x14xf16, {order = #NHWC}>)
            -> tensor<1x512x14x14xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
        : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<512x1x1x4xsi32> = #const.Content<dense<1>
        : tensor<512x1x1x4xsi32>>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverKernel",
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [512, 256, 3, 3],
        strides = [1, 1]
    } -> tensor<1x512x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x512x14x14xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-SAME:          [#const.SubView<[256, 0, 0, 0], [256, 1, 1, 4]>]
    // CHECK:       [[WEIGHTS2:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}>
    // CHECK-SAME:          [#const.Reorder<#NHWC>, #const.SubView<[256, 0, 0, 0], [256, 256, 3, 3]>]
    // CHECK:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-SAME:          [#const.SubView<[0, 0, 0, 0], [256, 1, 1, 4]>]
    // CHECK:       [[WEIGHTS1:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}>
    // CHECK-SAME:          [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [256, 256, 3, 3]>]

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:          rawFilterShape = [256, 256, 3, 3]
    // CHECK-SAME:          -> tensor<1x256x14x14xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:          rawFilterShape = [256, 256, 3, 3]
    // CHECK-SAME:          -> tensor<1x256x14x14xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV1]], [[CONV2]])

    // CHECK:       return [[CONCAT]] : tensor<1x512x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @SplitNCEEltwise
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x512x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x512x28x28xf16, {order = #NHWC}>
func @SplitNCEEltwise(
        %arg0: tensor<1x512x28x28xf16, {order = #NHWC}>,
        %arg1: tensor<1x512x28x28xf16, {order = #NHWC}>)
            -> tensor<1x512x28x28xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD"
    } -> tensor<1x512x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x512x28x28xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:       [[INPUT_0_0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>
    // CHECK:       [[INPUT_1_0:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[INPUT_0_0]], [[INPUT_1_0]])
    // CHECK-SAME:      {op_type = "ADD", tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:      -> tensor<1x256x28x28xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:       [[INPUT_0_1:%.+]] = VPU.Slice [[INPUT_0]] [0, 256, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>
    // CHECK:       [[INPUT_1_1:%.+]] = VPU.Slice [[INPUT_1]] [0, 256, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[INPUT_0_1]], [[INPUT_1_1]])
    // CHECK-SAME:      {op_type = "ADD", tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:      -> tensor<1x256x28x28xf16, {order = #NHWC}>

    // Concat
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_0]], [[ELTWISE_1]])
    // CHECK-SAME:      : tensor<1x256x28x28xf16, {order = #NHWC}>, tensor<1x256x28x28xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x512x28x28xf16, {order = #NHWC}>

    // return [[CONCAT]] : tensor<1x512x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @NoPrefetchingForEltwise
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x32x70x70xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x64x70x70xf16, {order = #NHWC}>
func @NoPrefetchingForEltwise(
        %arg0: tensor<1x32x70x70xf16, {order = #NHWC}>,
        %arg1: tensor<1x64x70x70xf16, {order = #NHWC}>)
            -> tensor<1x64x70x70xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = #const.Content<dense<1> : tensor<64x1x1x4xsi32>>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [64, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x64x70x70xf16, {order = #NHWC}>

    %1 = VPU.NCE.Eltwise(%0, %arg1) {
        op_type = "ADD"
    } -> tensor<1x64x70x70xf16, {order = #NHWC}>

    return %1 : tensor<1x64x70x70xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = #const.Content<dense<1>
    // CHECK:       [[WEIGHTS:%.+]]       = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00>

    // CHECK:       [[PARENT_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_0]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          -> tensor<1x64x70x70xf16, {order = #NHWC}>

    // Eltwise is not tiled for prefetching
    // CHECK-NOT:   VPU.Slice
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[PARENT_CONV]], [[INPUT_1]]) {op_type = "ADD"}
    // CHECK-SAME:          -> tensor<1x64x70x70xf16, {order = #NHWC}>

    // return [[ELTWISE]] : tensor<1x64x70x70xf16, {order = #NHWC}>
}

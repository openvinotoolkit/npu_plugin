// RUN: vpux-opt --split-input-file --prefetch-tiling --canonicalize %s | FileCheck %s


IE.MemoryResource 3200000 bytes of @CMX_NN

func @SplitOverOC(
        %input: tensor<1x32x100x100xf16>,
        %filter: tensor<128x32x3x3xf16>,
        %bias: tensor<1x128x1x1xf16>)
            -> tensor<1x128x100x100xf16> {
    %1 = IE.Convolution(%input, %filter, %bias) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x100x100xf16>, tensor<128x32x3x3xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x100x100xf16>
    return %1 : tensor<1x128x100x100xf16>
}

// CHECK-LABEL: func @SplitOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<128x32x3x3xf16>,
// CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x128x1x1xf16>

// Tile 0

// CHECK:       [[FILTER_TILE0:%.+]] = IE.Slice [[FILTER]] [0, 0, 0, 0] [64, 32, 3, 3]
// CHECK-SAME:      : tensor<128x32x3x3xf16> to tensor<64x32x3x3xf16>

// CHECK:       [[BIAS_TILE0:%.+]] = IE.Slice [[BIAS]] [0, 0, 0, 0] [1, 64, 1, 1]
// CHECK-SAME:      : tensor<1x128x1x1xf16> to tensor<1x64x1x1xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = IE.Convolution([[INPUT]], [[FILTER_TILE0]], [[BIAS_TILE0]])
// CHECK-SAME:          dilations = [1, 1]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
// CHECK-SAME:      -> tensor<1x64x100x100xf16>

// Tile 1

// CHECK:       [[FILTER_TILE1:%.+]] = IE.Slice [[FILTER]] [64, 0, 0, 0] [64, 32, 3, 3]
// CHECK-SAME:      : tensor<128x32x3x3xf16> to tensor<64x32x3x3xf16>

// CHECK:       [[BIAS_TILE1:%.+]] = IE.Slice [[BIAS]] [0, 64, 0, 0] [1, 64, 1, 1]
// CHECK-SAME:      : tensor<1x128x1x1xf16> to tensor<1x64x1x1xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = IE.Convolution([[INPUT]], [[FILTER_TILE1]], [[BIAS_TILE1]])
// CHECK-SAME:          dilations = [1, 1]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
// CHECK-SAME:      -> tensor<1x64x100x100xf16>

// Concat

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 64, 0, 0]
// CHECK-SAME:      -> tensor<1x128x100x100xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x128x100x100xf16>

// -----

IE.MemoryResource 400000 bytes of @CMX_NN

func @SplitOverH(
        %input: tensor<1x16x100x100xf16>)
            -> tensor<1x16x100x100xf16> {
    %1 = IE.MaxPool(%input) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x16x100x100xf16> -> tensor<1x16x100x100xf16>
    return %1 : tensor<1x16x100x100xf16>
}

// CHECK-LABEL: func @SplitOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x100x100xf16>

// Tile 0

// CHECK:       [[INPUT_TILE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 26, 100]
// CHECK-SAME:       : tensor<1x16x100x100xf16> to tensor<1x16x26x100xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = IE.MaxPool([[INPUT_TILE0]])
// CHECK-SAME:          kernel_size = [3, 3]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [0, 1]
// CHECK-SAME:          rounding_type = "FLOOR"
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:          tilingStrategy = [1, 1, 4, 1]
// CHECK-SAME:      -> tensor<1x16x25x100xf16>

// Tile 1

// CHECK:       [[INPUT_TILE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 24, 0] [1, 16, 27, 100]
// CHECK-SAME:      : tensor<1x16x100x100xf16> to tensor<1x16x27x100xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = IE.MaxPool([[INPUT_TILE1]])
// CHECK-SAME:          kernel_size = [3, 3]
// CHECK-SAME:          pads_begin = [0, 1]
// CHECK-SAME:          pads_end = [0, 1]
// CHECK-SAME:          rounding_type = "FLOOR"
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:          tilingStrategy = [1, 1, 4, 1]
// CHECK-SAME:      -> tensor<1x16x25x100xf16>

// Tile 2

// CHECK:       [[INPUT_TILE2:%.+]] = IE.Slice [[INPUT]] [0, 0, 49, 0] [1, 16, 27, 100]
// CHECK-SAME:      : tensor<1x16x100x100xf16> to tensor<1x16x27x100xf16>

// CHECK:       [[OUTPUT_TILE2:%.+]] = IE.MaxPool([[INPUT_TILE2]])
// CHECK-SAME:          kernel_size = [3, 3]
// CHECK-SAME:          pads_begin = [0, 1]
// CHECK-SAME:          pads_end = [0, 1]
// CHECK-SAME:          rounding_type = "FLOOR"
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:          tilingStrategy = [1, 1, 4, 1]
// CHECK-SAME:      -> tensor<1x16x25x100xf16>

// Tile 3

// CHECK:       [[INPUT_TILE3:%.+]] = IE.Slice [[INPUT]] [0, 0, 74, 0] [1, 16, 26, 100]
// CHECK-SAME:      : tensor<1x16x100x100xf16> to tensor<1x16x26x100xf16>

// CHECK:       [[OUTPUT_TILE3:%.+]] = IE.MaxPool([[INPUT_TILE3]])
// CHECK-SAME:          kernel_size = [3, 3]
// CHECK-SAME:          pads_begin = [0, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          rounding_type = "FLOOR"
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:          tilingStrategy = [1, 1, 4, 1]
// CHECK-SAME:      -> tensor<1x16x25x100xf16>

// Concat

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 0, 25, 0], [0, 0, 50, 0], [0, 0, 75, 0]
// CHECK-SAME:      -> tensor<1x16x100x100xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x16x100x100xf16>

// -----

!qElemType0 = type !quant.uniform<u8:f16:2, {0.1:127, 0.2:127, 0.3:127, 0.4:127, 0.5:127, 0.6:127, 0.7:127, 0.8:127}>
!qElemType1 = type !quant.uniform<u8:f16:2, {0.1:127, 0.2:127, 0.3:127, 0.4:127}>
!qElemType2 = type !quant.uniform<u8:f16:2, {0.5:127, 0.6:127, 0.7:127, 0.8:127}>

// 1x16x4x8xf16 + weights_table + act_window + profiling buffer
IE.MemoryResource 2400 bytes of @CMX_NN

// CHECK-LABEL: func @MultiAxesAndPerAxisQuant
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x8x8x!qElemType0>
func @MultiAxesAndPerAxisQuant(
        %input: tensor<1x32x8x8x!qElemType0>)
            -> tensor<1x32x8x8x!qElemType0> {
    %1 = IE.MaxPool(%input) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x32x8x8x!qElemType0> -> tensor<1x32x8x8x!qElemType0>
    return %1 : tensor<1x32x8x8x!qElemType0>
}

// Tile 0, 0

// CHECK:       [[INPUT_TILE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 4, 8]
// CHECK-SAME:      : tensor<1x32x8x8x!qElemType0> to tensor<1x16x4x8x!qElemType1>

// CHECK:       [[OUTPUT_TILE0:%.+]] = IE.MaxPool([[INPUT_TILE0]])
// CHECK-SAME:          kernel_size = [1, 1]
// CHECK-SAME:          pads_begin = [0, 0]
// CHECK-SAME:          pads_end = [0, 0]
// CHECK-SAME:          rounding_type = "FLOOR"
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:      -> tensor<1x16x4x8x!qElemType1>

// Tile 1, 0

// CHECK:       [[INPUT_TILE1:%.+]] = IE.Slice [[INPUT]] [0, 16, 0, 0] [1, 16, 4, 8]
// CHECK-SAME:      : tensor<1x32x8x8x!qElemType0> to tensor<1x16x4x8x!qElemType1>

// CHECK:       [[OUTPUT_TILE1:%.+]] = IE.MaxPool([[INPUT_TILE1]])
// CHECK-SAME:          kernel_size = [1, 1]
// CHECK-SAME:          pads_begin = [0, 0]
// CHECK-SAME:          pads_end = [0, 0]
// CHECK-SAME:          rounding_type = "FLOOR"
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:      -> tensor<1x16x4x8x!qElemType1>

// Tile 0, 1

// CHECK:       [[INPUT_TILE2:%.+]] = IE.Slice [[INPUT]] [0, 0, 4, 0] [1, 16, 4, 8]
// CHECK-SAME:      : tensor<1x32x8x8x!qElemType0> to tensor<1x16x4x8x!qElemType2>

// CHECK:       [[OUTPUT_TILE2:%.+]] = IE.MaxPool([[INPUT_TILE2]])
// CHECK-SAME:          kernel_size = [1, 1]
// CHECK-SAME:          pads_begin = [0, 0]
// CHECK-SAME:          pads_end = [0, 0]
// CHECK-SAME:          rounding_type = "FLOOR"
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:      -> tensor<1x16x4x8x!qElemType2>

// Tile 1, 1

// CHECK:       [[INPUT_TILE3:%.+]] = IE.Slice [[INPUT]] [0, 16, 4, 0] [1, 16, 4, 8]
// CHECK-SAME:      : tensor<1x32x8x8x!qElemType0> to tensor<1x16x4x8x!qElemType2>

// CHECK:       [[OUTPUT_TILE3:%.+]] = IE.MaxPool([[INPUT_TILE3]])
// CHECK-SAME:          kernel_size = [1, 1]
// CHECK-SAME:          pads_begin = [0, 0]
// CHECK-SAME:          pads_end = [0, 0]
// CHECK-SAME:          rounding_type = "FLOOR"
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:      -> tensor<1x16x4x8x!qElemType2>

// Concat

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 16, 0, 0], [0, 0, 4, 0], [0, 16, 4, 0]
// CHECK-SAME:      -> tensor<1x32x8x8x!qElemType0>

// CHECK:       return [[OUTPUT]] : tensor<1x32x8x8x!qElemType0>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @AvoidClusterTiling(%arg0: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<128x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]>
    %wt = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = #const.Content<dense<10> : tensor<128x1x1x4xsi32, {mem_space = @CMX_NN}>>

    %0 = VPU.NCE.ClusterTiling (
            %arg0 as %arg1: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights as %arg2: tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %wt as %arg3: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
                -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
                rawFilterShape = [128, 32, 3, 3],
                strides = [1, 1]
            } -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
      VPU.Yield %1
    }

    return %0 : tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// CHECK-LABEL:   @AvoidClusterTiling
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>

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

// -----

IE.MemoryResource 2400000 bytes of @CMX_NN

func @GenericTiling(
        %input: tensor<1x256x20x20xf16>,
        %filter1: tensor<256x256x3x3xf16>,
        %filter2: tensor<128x256x3x3xf16>,
        %bias1: tensor<1x256x1x1xf16>,
        %bias2: tensor<1x128x1x1xf16>)
            -> tensor<1x128x20x20xf16> {
    %1 = IE.Convolution(%input, %filter1, %bias1) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x256x20x20xf16>, tensor<256x256x3x3xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x20x20xf16>
    %2 = IE.And(%1, %1) {auto_broadcast = "NUMPY"} : tensor<1x256x20x20xf16>, tensor<1x256x20x20xf16> -> tensor<1x256x20x20xf16>
    %3 = IE.Convolution(%2, %filter2, %bias2) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x256x20x20xf16>, tensor<128x256x3x3xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x20x20xf16>
    return %3 : tensor<1x128x20x20xf16>
}

// CHECK-LABEL: func @GenericTiling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x256x20x20xf16>,
// CHECK-SAME:        [[FILTER1:%arg[0-9]]]: tensor<256x256x3x3xf16>,
// CHECK-SAME:        [[FILTER2:%arg[0-9]]]: tensor<128x256x3x3xf16>,
// CHECK-SAME:        [[BIAS1:%arg[0-9]]]: tensor<1x256x1x1xf16>,
// CHECK-SAME:        [[BIAS2:%arg[0-9]]]: tensor<1x128x1x1xf16>

// CHECK:       [[CONV_1:%.+]] = IE.Convolution([[INPUT]], [[FILTER1]], [[BIAS1]])
// CHECK-SAME:     {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
// CHECK-SAME:          -> tensor<1x256x20x20xf16>

// CHECK:       [[AND:%.+]] = IE.And([[CONV_1]], [[CONV_1]]) {auto_broadcast = "NUMPY"}
// CHECK-SAME:      tensor<1x256x20x20xf16>, tensor<1x256x20x20xf16>
// CHECK-SAME:          -> tensor<1x256x20x20xf16>

// Tile 0

// CHECK:       [[FILTER_TILE0:%.+]] = IE.Slice [[FILTER2]] [0, 0, 0, 0] [64, 256, 3, 3]
// CHECK-SAME:      tensor<128x256x3x3xf16> to tensor<64x256x3x3xf16>

// CHECK:       [[BIAS_TILE0:%.+]] = IE.Slice [[BIAS2]] [0, 0, 0, 0] [1, 64, 1, 1]
// CHECK-SAME:      tensor<1x128x1x1xf16> to tensor<1x64x1x1xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = IE.Convolution([[AND]], [[FILTER_TILE0]], [[BIAS_TILE0]])
// CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1], tilingStrategy = [1, 2, 1, 1]}
// CHECK-SAME:          -> tensor<1x64x20x20xf16>

// Tile 1

// CHECK:       [[FILTER_TILE1:%.+]] = IE.Slice [[FILTER2]] [64, 0, 0, 0] [64, 256, 3, 3]
// CHECK-SAME:      tensor<128x256x3x3xf16> to tensor<64x256x3x3xf16>

// CHECK:       [[BIAS_TILE1:%.+]] = IE.Slice [[BIAS2]] [0, 64, 0, 0] [1, 64, 1, 1]
// CHECK-SAME:      tensor<1x128x1x1xf16> to tensor<1x64x1x1xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = IE.Convolution([[AND]], [[FILTER_TILE1]], [[BIAS_TILE1]])
// CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1], tilingStrategy = [1, 2, 1, 1]}
// CHECK-SAME:          -> tensor<1x64x20x20xf16>

// Concat

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 64, 0, 0]
// CHECK-SAME:      -> tensor<1x128x20x20xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x128x20x20xf16>

func @AvoidMultiBranchesGenericTiling(
        %input: tensor<1x256x20x20xf16>,
        %filter: tensor<256x256x3x3xf16>,
        %filter2: tensor<128x256x3x3xf16>,
        %bias: tensor<1x256x1x1xf16>,
        %bias2: tensor<1x128x1x1xf16>)
            -> tensor<1x128x20x20xf16> {
    %1 = IE.Convolution(%input, %filter, %bias) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x256x20x20xf16>, tensor<256x256x3x3xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x20x20xf16>
    %2 = IE.And(%1, %1) {auto_broadcast = "NUMPY"} : tensor<1x256x20x20xf16>, tensor<1x256x20x20xf16> -> tensor<1x256x20x20xf16>
    %3 = IE.Convolution(%2, %filter2, %bias2) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x256x20x20xf16>, tensor<128x256x3x3xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x20x20xf16>
    %4 = IE.Convolution(%2, %filter2, %bias2) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x256x20x20xf16>, tensor<128x256x3x3xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x20x20xf16>
    return %4 : tensor<1x128x20x20xf16>
}

// CHECK-LABEL: func @AvoidMultiBranchesGenericTiling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x256x20x20xf16>,
// CHECK-SAME:        [[FILTER1:%arg[0-9]]]: tensor<256x256x3x3xf16>,
// CHECK-SAME:        [[FILTER2:%arg[0-9]]]: tensor<128x256x3x3xf16>,
// CHECK-SAME:        [[BIAS1:%arg[0-9]]]: tensor<1x256x1x1xf16>,
// CHECK-SAME:        [[BIAS2:%arg[0-9]]]: tensor<1x128x1x1xf16>

// CHECK:       [[CONV_1:%.+]] = IE.Convolution([[INPUT]], [[FILTER1]], [[BIAS1]])
// CHECK-SAME:     {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
// CHECK-SAME:          -> tensor<1x256x20x20xf16>

// CHECK:       [[AND:%.+]] = IE.And([[CONV_1]], [[CONV_1]]) {auto_broadcast = "NUMPY"}
// CHECK-SAME:      tensor<1x256x20x20xf16>, tensor<1x256x20x20xf16>
// CHECK-SAME:          -> tensor<1x256x20x20xf16>

// CHECK:       [[OUTPUT:%.+]] = IE.Convolution([[AND]], [[FILTER2]], [[BIAS2]])
// CHECK-SAME:     {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
// CHECK-SAME:          -> tensor<1x128x20x20xf16>

//CHECK:        return [[OUTPUT]] : tensor<1x128x20x20xf16>

// -----

IE.MemoryResource 3200000 bytes of @CMX_NN

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @SplitNCEConvOverOC(%arg0: tensor<1x32x100x100xf16, {order = #NHWC}>) -> tensor<1x128x100x100xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<128x1x1x4xsi32> = #const.Content<dense<1> : tensor<128x1x1x4xsi32>>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [128, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x128x100x100xf16, {order = #NHWC}>

    return %0 : tensor<1x128x100x100xf16, {order = #NHWC}>
}

// CHECK-LABEL:   @SplitNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {order = #NHWC}>

// CHECK:        [[WEIGHTS_TABLE_TILE1:%.+]] = const.Declare tensor<64x1x1x4xsi32> = #const.Content<dense<1>
// CHECK-SAME:      : tensor<128x1x1x4xsi32>, [#const.SubView<[64, 0, 0, 0], [64, 1, 1, 4]>]>

// CHECK:        [[FILTER_TILE1:%.+]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
// CHECK-SAME:      : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[64, 0, 0, 0], [64, 32, 3, 3]>]>

// CHECK:        [[WEIGHTS_TABLE_TILE0:%.+]] = const.Declare tensor<64x1x1x4xsi32> = #const.Content<dense<1>
// CHECK-SAME:      : tensor<128x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 4]>]>

// CHECK:        [[FILTER_TILE0:%.+]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00>
// CHECK-SAME:      : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [64, 32, 3, 3]>]>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]], [[WEIGHTS_TABLE_TILE0]])
// CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
// CHECK-SAME:          rawFilterShape = [64, 32, 3, 3],
// CHECK-SAME:          -> tensor<1x64x100x100xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[WEIGHTS_TABLE_TILE1]])
// CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
// CHECK-SAME:          rawFilterShape = [64, 32, 3, 3],
// CHECK-SAME:          -> tensor<1x64x100x100xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:          [0, 0, 0, 0], [0, 64, 0, 0]
// CHECK-SAME:          -> tensor<1x128x100x100xf16, {order = #NHWC}>

// CHECK:       return [[OUTPUT]] : tensor<1x128x100x100xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.MemoryResource 400000 bytes of @CMX_NN

func @SplitNCEPoolOverH(%arg0: tensor<1x16x100x100xf16, {order = #NHWC}>) -> tensor<1x16x100x100xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>
    %activation_window = const.Declare tensor<1x1x1x16xui8> = #const.Content<dense<1> : tensor<1x1x1x16xui8>>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 18 : i64,
        kernel_size = [3, 3],
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        strides = [1, 1]
    } -> tensor<1x16x100x100xf16, {order = #NHWC}>

    return %0 : tensor<1x16x100x100xf16, {order = #NHWC}>
}

// CHECK-LABEL: @SplitNCEPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x100x100xf16, {order = #NHWC}>)

// CHECK:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>
// CHECK-SAME:      = #const.Content<dense<1> : tensor<1x1x1x16xui8>>

// CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
// CHECK-SAME:      = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>

// CHECK:       [[INPUT_TILE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 26, 100]
// CHECK-SAME:      : tensor<1x16x100x100xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x16x26x100xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
// CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
// CHECK-SAME:      } -> tensor<1x16x25x100xf16, {order = #NHWC}>

// CHECK:       [[INPUT_TILE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 24, 0] [1, 16, 27, 100]
// CHECK-SAME:      : tensor<1x16x100x100xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x16x27x100xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
// CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK-SAME:      } -> tensor<1x16x25x100xf16, {order = #NHWC}>

// CHECK:       [[INPUT_TILE2:%.+]] = IE.Slice [[INPUT]] [0, 0, 49, 0] [1, 16, 27, 100]
// CHECK-SAME:      : tensor<1x16x100x100xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x16x27x100xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE2]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
// CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK-SAME:      } -> tensor<1x16x25x100xf16, {order = #NHWC}> 

// CHECK:       [[INPUT_TILE3:%.+]] = IE.Slice [[INPUT]] [0, 0, 74, 0] [1, 16, 26, 100]
// CHECK-SAME:      : tensor<1x16x100x100xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x16x26x100xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE3]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
// CHECK-SAME:      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
// CHECK-SAME:      } -> tensor<1x16x25x100xf16, {order = #NHWC}> 

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]]) {
// CHECK-SAME:      [0, 0, 0, 0], [0, 0, 25, 0], [0, 0, 50, 0], [0, 0, 75, 0]
// CHECK-SAME:      -> tensor<1x16x100x100xf16, {order = #NHWC}>

// CHECK:       return [[OUTPUT]] : tensor<1x16x100x100xf16, {order = #NHWC}>

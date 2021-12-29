// RUN: vpux-opt --split-input-file --isolated-tiling --canonicalize %s | FileCheck %s

IE.MemoryResource 1000000 bytes of @CMX_NN

func @InterpSplitOverH(
        %input1: tensor<1x32x128x128xf16>)
            -> tensor<1x32x512x512xf16> {

    %0 = const.Declare tensor<2xsi64> = #const.Content<dense<[512, 512]> : tensor<2xsi64>>
    %1 = const.Declare tensor<2xf32>  = #const.Content<dense<[4.000000e+00, 4.00000e+00]> : tensor<2xf32>>
    %2 = const.Declare tensor<2xsi64> = #const.Content<dense<[2, 3]> : tensor<2xsi64>>

    %3 = IE.Interpolate(%input1, %0, %1, %2) {
    attr = {antialias = false, coord_mode = "half_pixel", cube_coeff = -7.500000e-01, mode = "linear", nearest_mode = "round_prefer_floor", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "sizes"},
    operand_segment_sizes = dense<1> : vector<4xi32> } : tensor<1x32x128x128xf16>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x32x512x512xf16>

    return %3 : tensor<1x32x512x512xf16>
}


// CHECK-LABEL: func @SplitOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1024x14x14xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x1024x14x14xf16>

// Tile 0

// CHECK:       [[INPUT0_TILE0:%.+]] = IE.Slice [[INPUT1]] [0, 0, 0, 0] [1, 512, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16> to tensor<1x512x14x14xf16>

// CHECK:       [[INPUT1_TILE0:%.+]] = IE.Slice [[INPUT2]] [0, 0, 0, 0] [1, 512, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16> to tensor<1x512x14x14xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = IE.Add([[INPUT0_TILE0]], [[INPUT1_TILE0]])
// CHECK-SAME:      -> tensor<1x512x14x14xf16>

// Tile 1

// CHECK:       [[INPUT0_TILE1:%.+]] = IE.Slice [[INPUT1]] [0, 512, 0, 0] [1, 512, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16> to tensor<1x512x14x14xf16>

// CHECK:       [[INPUT1_TILE1:%.+]] = IE.Slice [[INPUT2]] [0, 512, 0, 0] [1, 512, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16> to tensor<1x512x14x14xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = IE.Add([[INPUT0_TILE1]], [[INPUT1_TILE1]])
// CHECK-SAME:      -> tensor<1x512x14x14xf16>

// Concat

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 512, 0, 0]
// CHECK-SAME:      -> tensor<1x1024x14x14xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x1024x14x14xf16>
// -----

<<<<<<< HEAD
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
=======
>>>>>>> cb1df7cf48 (trivial implementation of tilingbuilder interface)

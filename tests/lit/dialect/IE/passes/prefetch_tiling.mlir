// RUN: vpux-opt --split-input-file --prefetch-tiling --canonicalize %s | FileCheck %s


IERT.RunTimeResources
    availableMemory : {
        MemoryResource 3200000 bytes of "CMX_NN"
    }
    usedMemory : {
    }
    executors : {
    }

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

// CHECK:       [[FILTER_TILE0:%.+]] = IE.Slice [[FILTER]] [0, 0, 0, 0] [32, 32, 3, 3]
// CHECK-SAME:      : tensor<128x32x3x3xf16> to tensor<32x32x3x3xf16>

// CHECK:       [[BIAS_TILE0:%.+]] = IE.Slice [[BIAS]] [0, 0, 0, 0] [1, 32, 1, 1]
// CHECK-SAME:      : tensor<1x128x1x1xf16> to tensor<1x32x1x1xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = IE.Convolution([[INPUT]], [[FILTER_TILE0]], [[BIAS_TILE0]])
// CHECK-SAME:          dilations = [1, 1]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:      -> tensor<1x32x100x100xf16>

// Tile 1

// CHECK:       [[FILTER_TILE1:%.+]] = IE.Slice [[FILTER]] [32, 0, 0, 0] [32, 32, 3, 3]
// CHECK-SAME:      : tensor<128x32x3x3xf16> to tensor<32x32x3x3xf16>

// CHECK:       [[BIAS_TILE1:%.+]] = IE.Slice [[BIAS]] [0, 32, 0, 0] [1, 32, 1, 1]
// CHECK-SAME:      : tensor<1x128x1x1xf16> to tensor<1x32x1x1xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = IE.Convolution([[INPUT]], [[FILTER_TILE1]], [[BIAS_TILE1]])
// CHECK-SAME:          dilations = [1, 1]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:      -> tensor<1x32x100x100xf16>

// Tile 2

// CHECK:       [[FILTER_TILE2:%.+]] = IE.Slice [[FILTER]] [64, 0, 0, 0] [32, 32, 3, 3]
// CHECK-SAME:      : tensor<128x32x3x3xf16> to tensor<32x32x3x3xf16>

// CHECK:       [[BIAS_TILE2:%.+]] = IE.Slice [[BIAS]] [0, 64, 0, 0] [1, 32, 1, 1]
// CHECK-SAME:      : tensor<1x128x1x1xf16> to tensor<1x32x1x1xf16>

// CHECK:       [[OUTPUT_TILE2:%.+]] = IE.Convolution([[INPUT]], [[FILTER_TILE2]], [[BIAS_TILE2]])
// CHECK-SAME:          dilations = [1, 1]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:      -> tensor<1x32x100x100xf16>

// Tile 3

// CHECK:       [[FILTER_TILE3:%.+]] = IE.Slice [[FILTER]] [96, 0, 0, 0] [32, 32, 3, 3]
// CHECK-SAME:      : tensor<128x32x3x3xf16> to tensor<32x32x3x3xf16>

// CHECK:       [[BIAS_TILE3:%.+]] = IE.Slice [[BIAS]] [0, 96, 0, 0] [1, 32, 1, 1]
// CHECK-SAME:      : tensor<1x128x1x1xf16> to tensor<1x32x1x1xf16>

// CHECK:       [[OUTPUT_TILE3:%.+]] = IE.Convolution([[INPUT]], [[FILTER_TILE3]], [[BIAS_TILE3]])
// CHECK-SAME:          dilations = [1, 1]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:      -> tensor<1x32x100x100xf16>

// Concat

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]
// CHECK-SAME:      -> tensor<1x128x100x100xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x128x100x100xf16>

// -----

IERT.RunTimeResources
    availableMemory : {
        MemoryResource 400000 bytes of "CMX_NN"
    }
    usedMemory : {
    }
    executors : {
    }

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
// CHECK-SAME:      -> tensor<1x16x25x100xf16>

// Concat

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 0, 25, 0], [0, 0, 50, 0], [0, 0, 75, 0]
// CHECK-SAME:      -> tensor<1x16x100x100xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x16x100x100xf16>

// -----

IERT.RunTimeResources
    availableMemory : {
        MemoryResource 1000000 bytes of "CMX_NN"
    }
    usedMemory : {
    }
    executors : {
    }

func @SplitOverC(
        %input1: tensor<1x1024x14x14xf16>,
        %input2: tensor<1x1024x14x14xf16>)
            -> tensor<1x1024x14x14xf16> {
    %1 = IE.Add(%input1, %input2) { auto_broadcast = "NUMPY" } : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16> -> tensor<1x1024x14x14xf16>
    return %1 : tensor<1x1024x14x14xf16>
}

// CHECK-LABEL: func @SplitOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1024x14x14xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x1024x14x14xf16>

// Tile 0

// CHECK:       [[INPUT0_TILE0:%.+]] = IE.Slice [[INPUT1]] [0, 0, 0, 0] [1, 256, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16> to tensor<1x256x14x14xf16>

// CHECK:       [[INPUT1_TILE0:%.+]] = IE.Slice [[INPUT2]] [0, 0, 0, 0] [1, 256, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16> to tensor<1x256x14x14xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = IE.Add([[INPUT0_TILE0]], [[INPUT1_TILE0]])
// CHECK-SAME:      -> tensor<1x256x14x14xf16>

// Tile 1

// CHECK:       [[INPUT0_TILE1:%.+]] = IE.Slice [[INPUT1]] [0, 256, 0, 0] [1, 256, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16> to tensor<1x256x14x14xf16>

// CHECK:       [[INPUT1_TILE1:%.+]] = IE.Slice [[INPUT2]] [0, 256, 0, 0] [1, 256, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16> to tensor<1x256x14x14xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = IE.Add([[INPUT0_TILE1]], [[INPUT1_TILE1]])
// CHECK-SAME:      -> tensor<1x256x14x14xf16>


// Tile 2

// CHECK:       [[INPUT0_TILE2:%.+]] = IE.Slice [[INPUT1]] [0, 512, 0, 0] [1, 256, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16> to tensor<1x256x14x14xf16>

// CHECK:       [[INPUT1_TILE2:%.+]] = IE.Slice [[INPUT2]] [0, 512, 0, 0] [1, 256, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16> to tensor<1x256x14x14xf16>

// CHECK:       [[OUTPUT_TILE2:%.+]] = IE.Add([[INPUT0_TILE2]], [[INPUT1_TILE2]])
// CHECK-SAME:      -> tensor<1x256x14x14xf16>

// Tile 3

// CHECK:       [[INPUT0_TILE3:%.+]] = IE.Slice [[INPUT1]] [0, 768, 0, 0] [1, 256, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16> to tensor<1x256x14x14xf16>

// CHECK:       [[INPUT1_TILE3:%.+]] = IE.Slice [[INPUT2]] [0, 768, 0, 0] [1, 256, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16> to tensor<1x256x14x14xf16>

// CHECK:       [[OUTPUT_TILE3:%.+]] = IE.Add([[INPUT0_TILE3]], [[INPUT1_TILE3]])
// CHECK-SAME:      -> tensor<1x256x14x14xf16>

// Concat

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 256, 0, 0], [0, 512, 0, 0], [0, 768, 0, 0]
// CHECK-SAME:      -> tensor<1x1024x14x14xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x1024x14x14xf16>


// -----

!qElemType0 = type !quant.uniform<u8:f16:2, {0.1:127, 0.2:127, 0.3:127, 0.4:127, 0.5:127, 0.6:127, 0.7:127, 0.8:127}>
!qElemType1 = type !quant.uniform<u8:f16:2, {0.1:127, 0.2:127, 0.3:127, 0.4:127}>
!qElemType2 = type !quant.uniform<u8:f16:2, {0.5:127, 0.6:127, 0.7:127, 0.8:127}>

IERT.RunTimeResources
    availableMemory : {
        // 1x16x4x8xf16 + weights_table + act_window + profiling buffer
        MemoryResource 2400 bytes of "CMX_NN"
    }
    usedMemory : {
    }
    executors : {
    }

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

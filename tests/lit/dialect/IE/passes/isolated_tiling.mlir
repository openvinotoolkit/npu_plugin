// RUN: vpux-opt --split-input-file --isolated-tiling --canonicalize %s | FileCheck %s

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

// CHECK:       [[INPUT_TILE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 51, 100]
// CHECK-SAME:       : tensor<1x16x100x100xf16> to tensor<1x16x51x100xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = IE.MaxPool([[INPUT_TILE0]])
// CHECK-SAME:          kernel_size = [3, 3]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [0, 1]
// CHECK-SAME:          rounding_type = "FLOOR"
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:      -> tensor<1x16x50x100xf16>

// Tile 1

// CHECK:       [[INPUT_TILE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 49, 0] [1, 16, 51, 100]
// CHECK-SAME:      : tensor<1x16x100x100xf16> to tensor<1x16x51x100xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = IE.MaxPool([[INPUT_TILE1]])
// CHECK-SAME:          kernel_size = [3, 3]
// CHECK-SAME:          pads_begin = [0, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          rounding_type = "FLOOR"
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:      -> tensor<1x16x50x100xf16>

// Concat

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 0, 50, 0]
// CHECK-SAME:      -> tensor<1x16x100x100xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x16x100x100xf16>

// -----

IE.MemoryResource 1000000 bytes of @CMX_NN

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

IE.MemoryResource 1000000 bytes of @CMX_NN

func @EltwiseSameInput(
        %input: tensor<1x1024x14x14xf16>)
            -> tensor<1x1024x14x14xf16> {
    %1 = IE.And(%input, %input) { auto_broadcast = "NUMPY" } : tensor<1x1024x14x14xf16>, tensor<1x1024x14x14xf16> -> tensor<1x1024x14x14xf16>
    return %1 : tensor<1x1024x14x14xf16>
}

// CHECK-LABEL: func @EltwiseSameInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1024x14x14xf16>

// Tile 0

// CHECK:       [[INPUT_TILE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 14, 14]
// CHECK-SAME:       : tensor<1x1024x14x14xf16> to tensor<1x512x14x14xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = IE.And([[INPUT_TILE0]], [[INPUT_TILE0]])
// CHECK-SAME:      -> tensor<1x512x14x14xf16>

// Tile 1

// CHECK:       [[INPUT_TILE1:%.+]] = IE.Slice [[INPUT]] [0, 512, 0, 0] [1, 512, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16> to tensor<1x512x14x14xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = IE.And([[INPUT_TILE1]], [[INPUT_TILE1]])
// CHECK-SAME:      -> tensor<1x512x14x14xf16>

// Concat

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 512, 0, 0]
// CHECK-SAME:      -> tensor<1x1024x14x14xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x1024x14x14xf16>

// -----

!qElemType0 = type !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127,6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,8.5112804502952756E-4:127,6.7859559547244093E-4:127,6.4687653789370083E-4:127,7.1992648868110238E-4:127,9.9770853838582673E-4:127,8.1075833538385824E-4:127,6.2476931594488186E-4:127,6.8580447219488186E-4:127,6.7042553518700788E-4:127,0.001030388779527559:127,9.2225562869094487E-4:127,8.5881751353346455E-4:127,2.8330885519192912E-4:127,5.1423320620078736E-4:127,4.921259842519685E-4:127,7.3722779281496062E-4:127,7.8192282849409451E-4:127,6.9253275713582673E-4:127,9.9482498769685036E-4:127,7.9922413262795275E-4:127,6.7234790231299212E-4:127,5.6133120078740155E-4:127,8.1460306963582673E-4:127,6.1707984744094487E-4:127,8.4199680118110238E-4:127,3.7870632381889762E-4:127,3.5467673474409451E-4:127,3.1262495386318896E-4:127,3.020519346702756E-4:127,7.1944589689960632E-4:127,4.5992633489173228E-4:127,3.7293922244094487E-4:127,6.7619263656496062E-4:127,7.5452909694881886E-4:127,3.1166377030019684E-4:127,8.4103561761811026E-4:127,4.6521284448818896E-4:127,7.7086921751968508E-4:127,0.0014177457554133859:127,6.065068282480315E-4:127,6.5648837352362207E-4:127,5.0317959522637793E-4:127,6.382258858267716E-4:127,6.5889133243110238E-4:127,5.4066575418307088E-4:127,7.5116495447834642E-4:127,5.5796705831692912E-4:127,6.6946435162401575E-4:127,5.4835522268700788E-4:127,5.8055487204724415E-4:127,7.8048105314960632E-4:127,6.7042553518700788E-4:127,6.7138671875E-4:127,6.3582292691929129E-4:127}>
!qElemType2 = type !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType3 = type !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127,6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,8.5112804502952756E-4:127,6.7859559547244093E-4:127,6.4687653789370083E-4:127,7.1992648868110238E-4:127,9.9770853838582673E-4:127,8.1075833538385824E-4:127,6.2476931594488186E-4:127,6.8580447219488186E-4:127,6.7042553518700788E-4:127,0.001030388779527559:127,9.2225562869094487E-4:127,8.5881751353346455E-4:127,2.8330885519192912E-4:127,5.1423320620078736E-4:127,4.921259842519685E-4:127,7.3722779281496062E-4:127,7.8192282849409451E-4:127,6.9253275713582673E-4:127,9.9482498769685036E-4:127,7.9922413262795275E-4:127,6.7234790231299212E-4:127,5.6133120078740155E-4:127}>
!qElemType4 = type !quant.uniform<u8<0:254>:f16:0, {8.1460306963582673E-4:127,6.1707984744094487E-4:127,8.4199680118110238E-4:127,3.7870632381889762E-4:127,3.5467673474409451E-4:127,3.1262495386318896E-4:127,3.020519346702756E-4:127,7.1944589689960632E-4:127,4.5992633489173228E-4:127,3.7293922244094487E-4:127,6.7619263656496062E-4:127,7.5452909694881886E-4:127,3.1166377030019684E-4:127,8.4103561761811026E-4:127,4.6521284448818896E-4:127,7.7086921751968508E-4:127,0.0014177457554133859:127,6.065068282480315E-4:127,6.5648837352362207E-4:127,5.0317959522637793E-4:127,6.382258858267716E-4:127,6.5889133243110238E-4:127,5.4066575418307088E-4:127,7.5116495447834642E-4:127,5.5796705831692912E-4:127,6.6946435162401575E-4:127,5.4835522268700788E-4:127,5.8055487204724415E-4:127,7.8048105314960632E-4:127,6.7042553518700788E-4:127,6.7138671875E-4:127,6.3582292691929129E-4:127}>

IE.MemoryResource 240000 bytes of @CMX_NN

func @SplitQuantOverOC(
        %input: tensor<1x32x50x50x!qElemType0>,
        %filter: tensor<64x32x3x3x!qElemType1>)
            -> tensor<1x64x50x50x!qElemType2> {
    %1 = IE.Convolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x50x50x!qElemType0>, tensor<64x32x3x3x!qElemType1> -> tensor<1x64x50x50x!qElemType2>
    return %1 : tensor<1x64x50x50x!qElemType2>
}

// CHECK-LABEL: func @SplitQuantOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x50x50x!qElemType0>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<64x32x3x3x!qElemType1>

// Tile 0

// CHECK:       [[FILTER_TILE0:%.+]] = IE.Slice [[FILTER]] [0, 0, 0, 0] [32, 32, 3, 3]
// CHECK-SAME:      : tensor<64x32x3x3x!qElemType1> to tensor<32x32x3x3x!qElemType3>

// CHECK:       [[OUTPUT_TILE0:%.+]] = IE.Convolution([[INPUT]], [[FILTER_TILE0]])
// CHECK-SAME:          dilations = [1, 1]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:      -> tensor<1x32x50x50x!qElemType2>

// Tile 1

// CHECK:       [[FILTER_TILE1:%.+]] = IE.Slice [[FILTER]] [32, 0, 0, 0] [32, 32, 3, 3]
// CHECK-SAME:      : tensor<64x32x3x3x!qElemType1> to tensor<32x32x3x3x!qElemType4>

// CHECK:       [[OUTPUT_TILE1:%.+]] = IE.Convolution([[INPUT]], [[FILTER_TILE1]])
// CHECK-SAME:          dilations = [1, 1]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          strides = [1, 1]
// CHECK-SAME:      -> tensor<1x32x50x50x!qElemType2>

// Concat

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 32, 0, 0]
// CHECK-SAME:      -> tensor<1x64x50x50x!qElemType2>

// CHECK:       return [[OUTPUT]] : tensor<1x64x50x50x!qElemType2>

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

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @AvoidClusterTiling(%arg0: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<128x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = #const.Content<dense<10> : tensor<32x1x1x4xsi32, {mem_space = @CMX_NN}>>
    %activation_window = const.Declare tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}> = #const.Content<dense<1> : tensor<1x1x1x16xui8, {mem_space = @CMX_NN}>>

    %0 = VPU.NCE.ClusterTiling (
            %arg0 as %arg1: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights as %arg2: tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights_table as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
            %activation_window as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
                 -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
                              (activationWindow : %arg4 : )
                              (bias : #const.Content<dense<1.000000e+00> : tensor<1x128x1x1xf16>>) {
                pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
                rawFilterShape = [128, 32, 3, 3],
                strides = [1, 1],
                activation_window_channel_length = 44
            } : tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
                -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
      VPU.Yield %1
    }

    return %0 : tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// CHECK-LABEL:   @AvoidClusterTiling
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:        [[ACT_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
// CHECK:        [[WEIGHT_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
// CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:        [[CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling (
// CHECK-SAME:          %arg0 as %arg1: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:          [[WEIGHTS]] as %arg2: tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:          [[WEIGHT_TABLE]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
// CHECK-SAME:          [[ACT_WINDOW]] as %arg4: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
// CHECK-SAME:          -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:           [[NCE_CONV:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
// CHECK-SAME:              activationWindow : %arg4
// CHECK-SAME:              bias : #const.Content<dense<1.000000e+00> : tensor<1x128x1x1xf16>>
// CHECK-SAME:              activation_window_channel_length = 44
// CHECK-SAME:              pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
// CHECK-SAME:              strides = [1, 1]
// CHECK-SAME:              -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:           VPU.Yield [[NCE_CONV]]

// CHECK:         return [[CLUSTER_TILING]] : tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.MemoryResource 3200000 bytes of @CMX_NN

func @SplitNCEConvOverOC(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = #const.Content<dense<10> : tensor<16x1x1x4xsi32>>
    %activation_window = const.Declare tensor<1x1x1x16xui8, {order = #NCHW}> = #const.Content<dense<1> : tensor<1x1x1x16xui8>>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table)
        (activationWindow : %activation_window : )
        (bias : #const.Content<dense<1.000000e+00> : tensor<1x16x1x1xf16>>) {
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1]
    } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// CHECK-LABEL:   @SplitNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {order = #NHWC}>

// Weights tiles

// CHECK:        [[FILTER_TILE1:%.+]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}>
// CHECK-SAME:          : tensor<128x32x3x3xf16>,
// CHECK-SAME:          [#const.Reorder<#NHWC>, #const.SubView<[64, 0, 0, 0], [64, 32, 3, 3]>]>

// CHECK:        [[FILTER_TILE0:%.+]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}>
// CHECK-SAME:          : tensor<128x32x3x3xf16>,
// CHECK-SAME:          [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [64, 32, 3, 3]>]>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]])
// CHECK-SAME:          (bias : #const.Content<dense<1.000000e+00> : tensor<1x128x1x1xf16>,
// CHECK-SAME:                 [#const.SubView<[0, 0, 0, 0], [1, 64, 1, 1]>]>)
// CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
// CHECK-SAME:          rawFilterShape = [64, 32, 3, 3],
// CHECK-SAME:          -> tensor<1x64x100x100xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]])
// CHECK-SAME:          (bias : #const.Content<dense<1.000000e+00> : tensor<1x128x1x1xf16>,
// CHECK-SAME:                 [#const.SubView<[0, 64, 0, 0], [1, 64, 1, 1]>]>)
// CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
// CHECK-SAME:          rawFilterShape = [64, 32, 3, 3],
// CHECK-SAME:          -> tensor<1x64x100x100xf16, {order = #NHWC}>

// Concat

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:          [0, 0, 0, 0], [0, 64, 0, 0]
// CHECK-SAME:          -> tensor<1x128x100x100xf16, {order = #NHWC}>

// CHECK:       return [[OUTPUT]] : tensor<1x128x100x100xf16, {order = #NHWC}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.MemoryResource 400000 bytes of @CMX_NN

func @SplitNCEMaxPoolOverH(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = #const.Content<dense<10> : tensor<16x1x1x4xsi32>>
    %activation_window = const.Declare tensor<1x1x1x16xui8, {order = #NCHW}> = #const.Content<dense<1> : tensor<1x1x1x16xui8>>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 4 : i64,
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        strides = [1, 1]
    } -> tensor<1x16x1x4xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>
}

// CHECK-LABEL: @SplitNCEMaxPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x100x100xf16, {order = #NHWC}>)

// Tile 0

// CHECK:       [[INPUT_TILE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 51, 100]
// CHECK-SAME:      : tensor<1x16x100x100xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x16x51x100xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]]) {
// CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
// CHECK-SAME:      } -> tensor<1x16x50x100xf16, {order = #NHWC}>

// Tile 1

// CHECK:       [[INPUT_TILE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 49, 0] [1, 16, 51, 100]
// CHECK-SAME:      : tensor<1x16x100x100xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x16x51x100xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]]) {
// CHECK-SAME:      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
// CHECK-SAME:      } -> tensor<1x16x50x100xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 0, 50, 0]
// CHECK-SAME:      -> tensor<1x16x100x100xf16, {order = #NHWC}>

// CHECK:       return [[OUTPUT]] : tensor<1x16x100x100xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.MemoryResource 1000000 bytes of @CMX_NN

func @SplitNCEEltwiseAddSameInput(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = "ADD",
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64,
               lrelu_shift = 0 : i64,
               mode = "ADD"}
    } -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// CHECK-LABEL: @SplitNCEEltwiseAddSameInput
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1024x14x14xf16, {order = #NHWC}>

// Tile 0

// CHECK:       [[INPUT_TILE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x512x14x14xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Eltwise([[INPUT_TILE0]], [[INPUT_TILE0]]) {
// CHECK-SAME:      op_type = "ADD"
// CHECK-SAME:      } -> tensor<1x512x14x14xf16, {order = #NHWC}>

// CHECK:       [[INPUT_TILE1:%.+]] = IE.Slice [[INPUT]] [0, 512, 0, 0] [1, 512, 14, 14]
// CHECK-SAME:      : tensor<1x1024x14x14xf16, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x512x14x14xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Eltwise([[INPUT_TILE1]], [[INPUT_TILE1]]) {
// CHECK-SAME:      op_type = "ADD"
// CHECK-SAME:      } -> tensor<1x512x14x14xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 512, 0, 0]
// CHECK-SAME:      -> tensor<1x1024x14x14xf16, {order = #NHWC}>

// CHECK:       return [[OUTPUT]] : tensor<1x1024x14x14xf16, {order = #NHWC}>

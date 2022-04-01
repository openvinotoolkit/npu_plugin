// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX compilation-mode=DefaultHW" --adjust-memory-space %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @ConvNCEtoCMX(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1]
    } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_DDR:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[WEIGHTS_TABLE_DDR:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[IN_CMX:%.+]] = IE.Copy(%arg0) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[WEIGHTS_CMX:%.+]] = IE.Copy([[WEIGHTS_DDR]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<16x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = IE.Copy([[WEIGHTS_TABLE_DDR]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.Convolution([[IN_CMX]], [[WEIGHTS_CMX]], [[WEIGHTS_TABLE_CMX]])
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_DDR:%.+]] = IE.Copy([[OUT_CMX]])
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_DDR]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @DepthConvNCEtoCMX(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]>
    %activation_window = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> = #const.Content<dense<1> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]>

    %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %weights_table, %activation_window) {
        activation_window_channel_length = 44 : i64,
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        rawFilterShape = [16, 1, 4, 8],
        strides = [1, 1]
    } -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_DDR:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>
    // CHECK:       [[WEIGHTS_TABLE_DDR:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>
    // CHECK:       [[ACTIVATION_WINDOW_DDR:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}>

    // CHECK:       [[IN_CMX:%.+]] = IE.Copy(%arg0) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x16x40x80xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[WEIGHTS_CMX:%.+]] = IE.Copy([[WEIGHTS_DDR]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<16x1x4x8xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = IE.Copy([[WEIGHTS_TABLE_DDR]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[ACTIVATION_WINDOW_CMX:%.+]] = IE.Copy([[ACTIVATION_WINDOW_DDR]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x1x1x16xui8, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.DepthConvolution([[IN_CMX]], [[WEIGHTS_CMX]], [[WEIGHTS_TABLE_CMX]], [[ACTIVATION_WINDOW_CMX]])
    // CHECK-SAME:      activation_window_channel_length = 44 : i64,
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_DDR:%.+]] = IE.Copy([[OUT_CMX]])
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_DDR]] : tensor<1x16x37x73xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @MaxPoolNCEtoCMX(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]>
    %weights_table = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> = #const.Content<dense<1> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]>

    %0 = VPU.NCE.MaxPool(%arg0, %weights, %weights_table) {
        activation_window_channel_length = 4 : i64,
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        strides = [1, 1]
    } -> tensor<1x16x1x4xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_DDR:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>
    // CHECK:       [[WEIGHTS_TABLE_DDR:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}>

    // CHECK:       [[IN_CMX:%.+]] = IE.Copy(%arg0) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[WEIGHTS_CMX:%.+]] = IE.Copy([[WEIGHTS_DDR]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = IE.Copy([[WEIGHTS_TABLE_DDR]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x1x1x16xui8, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.MaxPool([[IN_CMX]], [[WEIGHTS_CMX]], [[WEIGHTS_TABLE_CMX]])
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_DDR:%.+]] = IE.Copy([[OUT_CMX]])
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_DDR]] : tensor<1x16x1x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @EltwiseAddNCEtoCMX(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>,
                         %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
                        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD"
    } -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[IN1_CMX:%.+]] = IE.Copy(%arg0) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[IN2_CMX:%.+]] = IE.Copy(%arg1) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.Eltwise([[IN1_CMX]], [[IN2_CMX]])
    // CHECK-SAME:      op_type = "ADD"
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_DDR:%.+]] = IE.Copy([[OUT_CMX]])
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_DDR]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @EltwiseAndSameInputsNCEtoCMX(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>)
                                  -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = "AND"
    } -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[IN_CMX:%.+]] = IE.Copy(%arg0) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.Eltwise([[IN_CMX]], [[IN_CMX]])
    // CHECK-SAME:      op_type = "AND"
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_DDR:%.+]] = IE.Copy([[OUT_CMX]])
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_DDR]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

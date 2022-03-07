// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB compilation-mode=DefaultHW" --adjust-memory-space %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @ConvNCEtoCMX(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %cst_w = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %cst_wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]>

    %0 = VPU.NCE.Convolution(%arg0, %cst_w, %cst_wt) {
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1]
    } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[CST_W:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CST_WT:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = IE.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = IE.Copy([[CST_W]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = IE.Copy([[CST_WT]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]], [[VAL1]], [[VAL2]])
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = IE.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @DepthConvNCEtoCMX(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %cst_w = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]>
    %cst_wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]>
    %cst_aw = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> = #const.Content<dense<1> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]>

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_w, %cst_wt, %cst_aw) {
        activation_window_channel_length = 44 : i64,
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        rawFilterShape = [16, 1, 4, 8],
        strides = [1, 1]
    } -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       [[CST_W:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>
    // CHECK:       [[CST_WT:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>
    // CHECK:       [[CST_AW:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = IE.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x40x80xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = IE.Copy([[CST_W]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = IE.Copy([[CST_WT]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL3:%.+]] = IE.Copy([[CST_AW]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]], [[VAL2]], [[VAL3]])
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL5:%.+]] = IE.Copy([[VAL4]])
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]] : tensor<1x16x37x73xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @MaxPoolNCEtoCMX(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %cst_wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]>
    %cst_aw = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> = #const.Content<dense<1> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]>

    %0 = VPU.NCE.MaxPool(%arg0, %cst_wt, %cst_aw) {
        activation_window_channel_length = 4 : i64,
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        strides = [1, 1]
    } -> tensor<1x16x1x4xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       [[CST_WT:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>
    // CHECK:       [[CST_AW:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = IE.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = IE.Copy([[CST_WT]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = IE.Copy([[CST_AW]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.MaxPool([[VAL0]], [[VAL1]], [[VAL2]])
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = IE.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x16x1x4xf16, {order = #NHWC}>
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

    // CHECK:       [[VAL0:%.+]] = IE.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Eltwise([[VAL0]], [[VAL1]])
    // CHECK-SAME:      op_type = "ADD"
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = IE.Copy([[VAL2]])
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[VAL3]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @EltwiseAndSameInputsNCEtoCMX(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>)
                                  -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = "AND"
    } -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = IE.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Eltwise([[VAL0]], [[VAL0]])
    // CHECK-SAME:      op_type = "AND"
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL2:%.+]] = IE.Copy([[VAL1]])
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[VAL2]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

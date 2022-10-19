// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --cmx-concat --canonicalize %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CaseWithoutChildTiling
module @CaseWithoutChildTiling {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x144x32x32xf16, {order = #NHWC}>
        DataInfo "filter" : tensor<48x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        DataInfo "weightsTable" : tensor<48x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NCHW}>
        DataInfo "activationWindow" : tensor<1x1x1x16xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
        DataInfo "weightsTableMaxPool" : tensor<144x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x144x32x32xf16, {order = #NHWC}>
    }

func @main(%input: tensor<1x144x32x32xf16, {order = #NHWC}>,
           %filter: tensor<48x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
           %weightsTable: tensor<48x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
           %activationWindow: tensor<1x1x1x16xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
           %weightsTableMaxPool: tensor<144x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NCHW}>)
           -> tensor<1x144x32x32xf16, {order = #NHWC}> {

    // Create a concat subgraph with three input tiles and one output user

    // Concat input tile 1
    %0 = VPU.Slice %input [0, 0, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>
    %1 = VPU.Copy(%0) {out_mem_space = [@CMX_NN, 0]} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    %2 = VPU.NCE.DepthConvolution(%1, %filter, %weightsTable, %activationWindow)
        {activation_window_channel_length = 18 : i64,
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"},
        rawFilterShape = [48, 1, 3, 3],
        strides = [1, 1]}
        -> tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // NCE copy-out to concatinate in DDR
    %3 = VPU.Copy(%2) : tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>

    // Concat input tile 2
    %4 = VPU.Slice %input [0, 48, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>
    %5 = VPU.Copy(%4) {out_mem_space = [@CMX_NN, 0]} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    %6 = VPU.NCE.DepthConvolution(%5, %filter, %weightsTable, %activationWindow)
        {activation_window_channel_length = 18 : i64,
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"},
        rawFilterShape = [48, 1, 3, 3],
        strides = [1, 1]}
        -> tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // NCE copy-out to concatinate in DDR
    %7 = VPU.Copy(%6) : tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>

    // Concat input tile 3
    %8 = VPU.Slice %input [0, 96, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>
    %9 = VPU.Copy(%8) {out_mem_space = [@CMX_NN, 0]} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    %10 = VPU.NCE.DepthConvolution(%9, %filter, %weightsTable, %activationWindow)
        {activation_window_channel_length = 18 : i64,
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64},
        name = "IE.Clamp"}, rawFilterShape = [48, 1, 3, 3],
        strides = [1, 1]}
        -> tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // NCE copy-out to concatinate in DDR
    %11 = VPU.Copy(%10) : tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>

    // Concat inputs are in DDR and Concat output is in DDR
    %12 = VPU.Concat(%3, %7, %11) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]]} : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x144x32x32xf16, {order = #NHWC}>

    // Concat result copy-in for NCE user
    %13 = VPU.Copy(%12) {out_mem_space = [@CMX_NN, 0]} : tensor<1x144x32x32xf16, {order = #NHWC}> -> tensor<1x144x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    %14 = VPU.NCE.MaxPool(%13, %weightsTableMaxPool, %activationWindow) {
            activation_window_channel_length = 4 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1], kernel_size = [1, 1]
        } -> tensor<1x144x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    %15 = VPU.Copy(%14) : tensor<1x144x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x144x32x32xf16, {order = #NHWC}>

    return %15 : tensor<1x144x32x32xf16, {order = #NHWC}>

    // the below checks that the concat is occuring in NNCMX and the copy operations
    // arround the concat are not used by the concat producer and consumer operations

    // input to the first tile copy-in (DDR->NNCMX) for activation and weights
    // CHECK:       [[VAL0:%.+]] = VPU.Slice %arg0
    // CHECK-SAME:      [0, 0, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[VAL0]])

    // first tile of NCE task
    // CHECK:       [[VAL2:%.+]] = VPU.NCE.DepthConvolution([[VAL1]], %arg1, %arg2, %arg3)

    // no copy-out to concatinate in DDR
    // CHECK-NOT:   VPU.Copy

    // input to the second tile copy-in (DDR->NNCMX) for activation and weights
    // CHECK:       [[VAL3:%.+]] = VPU.Slice %arg0
    // CHECK-SAME:      [0, 48, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])

    // second tile of NCE task
    // CHECK:       [[VAL5:%.+]] = VPU.NCE.DepthConvolution([[VAL4]], %arg1, %arg2, %arg3)

    // no copy-out to concatinate in DDR
    // CHECK-NOT:   VPU.Copy

    // input to the third tile copy-in (DDR->NNCMX) for activation and weights
    // CHECK:       [[VAL6:%.+]] = VPU.Slice %arg0
    // CHECK-SAME:      [0, 96, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:       [[VAL7:%.+]] = VPU.Copy([[VAL6]])

    // third tile of NCE task
    // CHECK:       [[VAL8:%.+]] = VPU.NCE.DepthConvolution([[VAL7]], %arg1, %arg2, %arg3)

    // no copy-out to concatinate in DDR
    // CHECK-NOT:   VPU.Copy

    // no copy-out (NNCMX->DDR) operations to concatinate in DDR
    // Concat in NNCMX using results of NCE tiles in NNCMX
    // CHECK:       [[VAL9:%.+]] = VPU.Concat([[VAL2]], [[VAL5]], [[VAL8]])

    // no Concat buffer copy-in to NNCMX
    // CHECK-NOT:   VPU.Copy

    // user of the concat uses result of concat without intermediate copy operation
    // CHECK:       [[VAL10:%.+]] = VPU.NCE.MaxPool([[VAL9]], %arg4, %arg3)
    // CHECK:       [[VAL11:%.+]] = VPU.Copy([[VAL10]])

    // CHECK:       return [[VAL11:%.+]] : tensor<1x144x32x32xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CaseWithChildTiling
module @CaseWithChildTiling {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x144x32x32xf16, {order = #NHWC}>
        DataInfo "filter" : tensor<48x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        DataInfo "weightsTable" : tensor<48x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NCHW}>
        DataInfo "activationWindow" : tensor<1x1x1x16xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
        DataInfo "filterCons" : tensor<144x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        DataInfo "weightsTableCons" : tensor<144x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    }
    outputsInfo : {
        DataInfo "prob1" : tensor<1x144x16x32xf16, {order = #NHWC}>
        DataInfo "prob2" : tensor<1x144x16x32xf16, {order = #NHWC}>
    }

func @main(%input: tensor<1x144x32x32xf16, {order = #NHWC}>,
           %filter: tensor<48x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
           %weightsTable: tensor<48x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
           %activationWindow: tensor<1x1x1x16xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
           %filterCons: tensor<144x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
           %weightsTableCons: tensor<144x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NCHW}>)
           -> (tensor<1x144x16x32xf16, {order = #NHWC}>, tensor<1x144x16x32xf16, {order = #NHWC}>) {

    // Create a concat subgraph with three input tiles and two output users
    
    // Concat input tile 1
    %0 = VPU.Slice %input [0, 0, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>
    %1 = VPU.Copy(%0) {out_mem_space = [@CMX_NN, 0]} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    %2 = VPU.NCE.DepthConvolution(%1, %filter, %weightsTable, %activationWindow)
        {activation_window_channel_length = 18 : i64,
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"},
        rawFilterShape = [48, 1, 3, 3],
        strides = [1, 1]}
        -> tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // NCE copy-out to concatinate in DDR
    %3 = VPU.Copy(%2) : tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>

    // Concat input tile 2
    %4 = VPU.Slice %input [0, 48, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>
    %5 = VPU.Copy(%4) {out_mem_space = [@CMX_NN, 0]} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    %6 = VPU.NCE.DepthConvolution(%5, %filter, %weightsTable, %activationWindow)
        {activation_window_channel_length = 18 : i64,
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"},
        rawFilterShape = [48, 1, 3, 3],
        strides = [1, 1]}
        -> tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // NCE copy-out to concatinate in DDR
    %7 = VPU.Copy(%6) : tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>

    // Concat input tile 3
    %8 = VPU.Slice %input [0, 96, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>
    %9 = VPU.Copy(%8) {out_mem_space = [@CMX_NN, 0]} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    %10 = VPU.NCE.DepthConvolution(%9, %filter, %weightsTable, %activationWindow)
        {activation_window_channel_length = 18 : i64,
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64},
        name = "IE.Clamp"}, rawFilterShape = [48, 1, 3, 3],
        strides = [1, 1]}
        -> tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // NCE copy-out to concatinate in DDR
    %11 = VPU.Copy(%10) : tensor<1x48x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>

    // Concat inputs are in DDR and Concat output is in DDR
    %12 = VPU.Concat(%3, %7, %11) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]]} : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x144x32x32xf16, {order = #NHWC}>

    %13 = VPU.Slice %12 [0, 0, 0, 0] [1, 144, 16, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x144x16x32xf16, {order = #NHWC}>
    // Concat slice result copy-in for NCE user
    %14 = VPU.Copy(%13) {out_mem_space = [@CMX_NN, 0]} : tensor<1x144x16x32xf16, {order = #NHWC}> -> tensor<1x144x16x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %15 = VPU.NCE.DepthConvolution(%14, %filterCons, %weightsTableCons, %activationWindow) {activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, rawFilterShape = [144, 1, 3, 3], strides = [1, 1]} -> tensor<1x144x16x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %16 = VPU.Copy(%15) : tensor<1x144x16x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x144x16x32xf16, {order = #NHWC}>

    %17 = VPU.Slice %12 [0, 144, 0, 0] [1, 144, 16, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x144x16x32xf16, {order = #NHWC}>
    // Concat slice result copy-in for NCE user
    %18 = VPU.Copy(%17) {out_mem_space = [@CMX_NN, 0]} : tensor<1x144x16x32xf16, {order = #NHWC}> -> tensor<1x144x16x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %19 = VPU.NCE.DepthConvolution(%18, %filterCons, %weightsTableCons, %activationWindow) {activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, rawFilterShape = [144, 1, 3, 3], strides = [1, 1]} -> tensor<1x144x16x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %20 = VPU.Copy(%19) : tensor<1x144x16x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x144x16x32xf16, {order = #NHWC}>

    return %16, %20 : tensor<1x144x16x32xf16, {order = #NHWC}>, tensor<1x144x16x32xf16, {order = #NHWC}>

    // the below checks that the concat is occuring in NNCMX and the copy operations
    // arround the concat are not used by the concat producer and consumer operations

    // input to the first tile copy-in (DDR->NNCMX) for activation and weights
    // CHECK:       [[VAL0:%.+]] = VPU.Slice %arg0
    // CHECK-SAME:      [0, 0, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[VAL0]])

    // first tile of NCE task
    // CHECK:       [[VAL2:%.+]] = VPU.NCE.DepthConvolution([[VAL1]], %arg1, %arg2, %arg3)

    // no copy-out to concatinate in DDR
    // CHECK-NOT:   VPU.Copy

    // input to the second tile copy-in (DDR->NNCMX) for activation and weights
    // CHECK:       [[VAL3:%.+]] = VPU.Slice %arg0
    // CHECK-SAME:      [0, 48, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])

    // second tile of NCE task
    // CHECK:       [[VAL5:%.+]] = VPU.NCE.DepthConvolution([[VAL4]], %arg1, %arg2, %arg3)

    // no copy-out to concatinate in DDR
    // CHECK-NOT:   VPU.Copy

    // input to the third tile copy-in (DDR->NNCMX) for activation and weights
    // CHECK:       [[VAL6:%.+]] = VPU.Slice %arg0
    // CHECK-SAME:      [0, 96, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:       [[VAL7:%.+]] = VPU.Copy([[VAL6]])

    // third tile of NCE task
    // CHECK:       [[VAL8:%.+]] = VPU.NCE.DepthConvolution([[VAL7]], %arg1, %arg2, %arg3)

    // no copy-out to concatinate in DDR
    // CHECK-NOT:   VPU.Copy

    // no copy-out (NNCMX->DDR) operations to concatinate in DDR
    // Concat in NNCMX using results of NCE tiles in NNCMX
    // CHECK:       [[VAL9:%.+]] = VPU.Concat([[VAL2]], [[VAL5]], [[VAL8]])

    // no Concat buffer copy-in to NNCMX
    // CHECK-NOT:   VPU.Copy

    // users of concat which use part of the master buffer through slices
    // concat partial buffer user slicing output of concat in NNCMX
    // CHECK:       [[VAL10:%.+]] = VPU.Slice [[VAL9]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 144, 16, 32] : tensor<1x144x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> to tensor<1x144x16x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // no Concat buffer slice copy-in to NNCMX
    // CHECK-NOT:   VPU.Copy

    // concat user reading from NNCMX slice without intermediate copy operation
    // CHECK:       [[VAL11:%.+]] = VPU.NCE.DepthConvolution([[VAL10]], %arg4, %arg5, %arg3)
    // copy-out
    // CHECK:       [[VAL12:%.+]] = VPU.Copy([[VAL11]])

    // user of second part of concat master buffer
    // CHECK:       [[VAL13:%.+]] = VPU.Slice [[VAL9]]
    // CHECK-SAME:      [0, 144, 0, 0] [1, 144, 16, 32] : tensor<1x144x32x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> to tensor<1x144x16x32xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // no Concat buffer slice copy-in to NNCMX
    // CHECK-NOT:   VPU.Copy

    // user reading from NNCMX without intermediate copy operation
    // CHECK:       [[VAL14:%.+]] = VPU.NCE.DepthConvolution([[VAL13]], %arg4, %arg5, %arg3)
    // copy-out
    // CHECK:       [[VAL15:%.+]] = VPU.Copy([[VAL14]])

    // CHECK:       return [[VAL12:%.+]], [[VAL15:%.+]] : tensor<1x144x16x32xf16, {order = #NHWC}>, tensor<1x144x16x32xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Distributed = type !VPU.DistributedTensor<
    1x144x32x32xf16, #NHWC, @CMX_NN, {
   mode = "DUPLICATED",
    num_clusters = 4
}>

!DistributedTile = type !VPU.DistributedTensor<
    1x48x32x32xf16, #NHWC, @CMX_NN, {
   mode = "DUPLICATED",
    num_clusters = 4
}>

!DistributedTileOutput = type !VPU.DistributedTensor<
    1x48x32x32xf16, #NHWC, @CMX_NN, {
   mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

module @CaseWithNceClusterTilingWithoutChildTiling {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x144x32x32xf16, {order = #NHWC}>
        DataInfo "filter" : tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
        DataInfo "weightsTable" : tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
        DataInfo "maxPoolWeightsTable" : tensor<144x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
        DataInfo "activationWindow" : tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x144x32x32xf16, {order = #NHWC}>
    }

func @main(%input: tensor<1x144x32x32xf16, {order = #NHWC}>,
           %filter: tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
           %weightsTable: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
           %maxPoolWeightsTable: tensor<144x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
           %activationWindow: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
           -> tensor<1x144x32x32xf16, {order = #NHWC}> {

    // Create a concat subgraph with three input tiles and one output user
    
    // Concat input tile 1
    %0 = VPU.Slice %input [0, 0, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>

    %1 = VPU.NCE.ClusterTiling (%0 as %arg0: tensor<1x48x32x32xf16, {order = #NHWC}>) -> !DistributedTile {
        %16 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %16
    }

    %2 = VPU.NCE.ClusterTiling (%1 as %arg0: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %filter as %arg1: tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable as %arg2: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !DistributedTileOutput {
        %16 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2, %arg3)
            {activation_window_channel_length = 18 : i64,
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
            post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"},
            rawFilterShape = [48, 1, 3, 3],
            strides = [1, 1]}
            -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %16
    }

    %3 = VPU.NCE.ClusterTiling (%2 as %arg0: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        %16 = VPU.Copy(%arg0) : tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        VPU.Yield %16
    }

    // Concat input tile 2
    %4 = VPU.Slice %input [0, 48, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>

    %5 = VPU.NCE.ClusterTiling (%4 as %arg0: tensor<1x48x32x32xf16, {order = #NHWC}>) -> !DistributedTile {
        %16 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %16
    }

    %6 = VPU.NCE.ClusterTiling (
        %5 as %arg0: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %filter as %arg1: tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable as %arg2: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !DistributedTileOutput {
        %16 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2, %arg3)
            {activation_window_channel_length = 18 : i64,
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
            post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"},
            rawFilterShape = [48, 1, 3, 3],
            strides = [1, 1]}
            -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %16
    }

    %7 = VPU.NCE.ClusterTiling (%6 as %arg0: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        %16 = VPU.Copy(%arg0) : tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        VPU.Yield %16
    }

    // Concat input tile 3
    %8 = VPU.Slice %input [0, 96, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>

    %9 = VPU.NCE.ClusterTiling (%8 as %arg0: tensor<1x48x32x32xf16, {order = #NHWC}>) -> !DistributedTile {
        %16 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %16
    }

    %10 = VPU.NCE.ClusterTiling (
        %9 as %arg0: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %filter as %arg1: tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable as %arg2: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !DistributedTileOutput {
        %16 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2, %arg3)
            {activation_window_channel_length = 18 : i64,
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
            post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64},
            name = "IE.Clamp"}, rawFilterShape = [48, 1, 3, 3],
            strides = [1, 1]}
            -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %16
    }

    %11 = VPU.NCE.ClusterTiling (%10 as %arg0: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        %16 = VPU.Copy(%arg0) : tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        VPU.Yield %16
    }

    %12 = VPU.Concat(%3, %7, %11) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]]} : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x144x32x32xf16, {order = #NHWC}>

    %13 = VPU.NCE.ClusterTiling (%12 as %arg0: tensor<1x144x32x32xf16, {order = #NHWC}>) -> !Distributed {
        %16 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x144x32x32xf16, {order = #NHWC}> -> tensor<1x144x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %16
    }

    %14 = VPU.NCE.ClusterTiling (
        %13 as %arg0: tensor<1x144x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %maxPoolWeightsTable as %arg1: tensor<144x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow as %arg2: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !Distributed {
        %16 = VPU.NCE.MaxPool(%arg0, %arg1, %arg2) {
                activation_window_channel_length = 4 : i64,
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> tensor<1x144x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %16
    }

    %15 = VPU.NCE.ClusterTiling (%14 as %arg0: tensor<1x144x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x144x32x32xf16, {order = #NHWC}> {
        %16 = VPU.Copy(%arg0) : tensor<1x144x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x144x32x32xf16, {order = #NHWC}>
        VPU.Yield %16
    }

    return %15 : tensor<1x144x32x32xf16, {order = #NHWC}>

}
}

// CHECK-LABEL: @CaseWithNceClusterTilingWithoutChildTiling

// Tile 0
// CHECK:       [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 48, 32, 32]

// CHECK:       [[COPY0:%.+]]  = VPU.NCE.ClusterTiling ([[SLICE0]] as %arg5: tensor<1x48x32x32xf16, {order = #NHWC}>)
// CHECK-SAME:      -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
// CHECK:           [[COPY0_INNER:%.+]] = VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x48x32x32xf16, {order = #NHWC}>
// CHECK-SAME:              -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[NCE0:%.+]]   = VPU.NCE.ClusterTiling ([[COPY0]] as %arg5: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
// CHECK:           [[NCE0_INNER:%.+]] = VPU.NCE.DepthConvolution(%arg5, %arg6, %arg7, %arg8)
// CHECK-SAME:              -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// Tile 1
// CHECK:       [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 48, 0, 0] [1, 48, 32, 32]

// CHECK:       [[COPY1:%.+]]  = VPU.NCE.ClusterTiling ([[SLICE1]] as %arg5: tensor<1x48x32x32xf16, {order = #NHWC}>)
// CHECK-SAME:      -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
// CHECK:           [[COPY1_INNER:%.+]] = VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x48x32x32xf16, {order = #NHWC}>
// CHECK-SAME:              -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[NCE1:%.+]]   = VPU.NCE.ClusterTiling ([[COPY1]] as %arg5: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
// CHECK:           [[NCE1_INNER:%.+]] = VPU.NCE.DepthConvolution(%arg5, %arg6, %arg7, %arg8)
// CHECK-SAME:              -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// Tile 2
// CHECK:       [[SLICE2:%.+]] = VPU.Slice %arg0 [0, 96, 0, 0] [1, 48, 32, 32]

// CHECK:       [[COPY2:%.+]]  = VPU.NCE.ClusterTiling ([[SLICE2]] as %arg5: tensor<1x48x32x32xf16, {order = #NHWC}>)
// CHECK-SAME:      -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
// CHECK:           [[COPY2_INNER:%.+]] = VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x48x32x32xf16, {order = #NHWC}>
// CHECK-SAME:              -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[NCE2:%.+]]   = VPU.NCE.ClusterTiling ([[COPY2]] as %arg5: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
// CHECK:           [[NCE1_INNER:%.+]] = VPU.NCE.DepthConvolution(%arg5, %arg6, %arg7, %arg8)
// CHECK-SAME:              -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// Concat
// CHECK:       [[CONCAT:%.+]] = VPU.Concat([[NCE0]], [[NCE1]], [[NCE2]])
// CHECK-SAME:      !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
// CHECK-SAME:      !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
// CHECK-SAME:      !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x144x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

// CHECK:       [[CAST:%.+]] = VPU.DistributedCast([[CONCAT]] :
// CHECK-SAME:      !VPU.DistributedTensor<1x144x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x144x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

// CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.ClusterTiling
// CHECK-SAME:      [[CAST]] as %arg5: tensor<1x144x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:          -> !VPU.DistributedTensor<1x144x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
// CHECK:           [[MAXPOOL_INNER:%.+]] = VPU.NCE.MaxPool(%arg5, %arg6, %arg7)
// CHECK-SAME:          activation_window_channel_length = 4 : i64,
// CHECK-SAME:          kernel_size = [1, 1],
// CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, strides = [1, 1]
// CHECK-SAME:              -> tensor<1x144x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}

// CHECK:       [[COPY_OUT:%.+]] = VPU.NCE.ClusterTiling
// CHECK-SAME:      ([[MAXPOOL]] as %arg5: tensor<1x144x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK-SAME:          -> tensor<1x144x32x32xf16, {order = #NHWC}>
// CHECK:           [[COPY_OUT_INNER:%.+]] = VPU.Copy(%arg5)
// CHECK-SAME:          : tensor<1x144x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:              -> tensor<1x144x32x32xf16, {order = #NHWC}>

// CHECK:       return [[COPY_OUT]] : tensor<1x144x32x32xf16, {order = #NHWC}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Distributed = type !VPU.DistributedTensor<
    1x144x16x32xf16, #NHWC, @CMX_NN, {
   mode = "DUPLICATED",
    num_clusters = 4
}>

!DistributedTile = type !VPU.DistributedTensor<
    1x48x32x32xf16, #NHWC, @CMX_NN, {
   mode = "DUPLICATED",
    num_clusters = 4
}>

!DistributedTileOutput = type !VPU.DistributedTensor<
    1x48x32x32xf16, #NHWC, @CMX_NN, {
   mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

module @CaseWithNceClusterTilingWithChildTiling {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x144x32x32xf16, {order = #NHWC}>
        DataInfo "filter" : tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
        DataInfo "weightsTable" : tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
        DataInfo "maxPoolWeightsTable" : tensor<144x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
        DataInfo "activationWindow" : tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    }
    outputsInfo : {
        DataInfo "prob1" : tensor<1x144x16x32xf16, {order = #NHWC}>
        DataInfo "prob2" : tensor<1x144x16x32xf16, {order = #NHWC}>
    }

func @main(%input: tensor<1x144x32x32xf16, {order = #NHWC}>,
           %filter: tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
           %weightsTable: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
           %maxPoolWeightsTable: tensor<144x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
           %activationWindow: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
           -> (tensor<1x144x16x32xf16, {order = #NHWC}>, tensor<1x144x16x32xf16, {order = #NHWC}>) {

    // Create a concat subgraph with three input tiles and two output users

    // Concat input tile 1
    %0 = VPU.Slice %input [0, 0, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>

    %1 = VPU.NCE.ClusterTiling (%0 as %arg0: tensor<1x48x32x32xf16, {order = #NHWC}>) -> !DistributedTile {
        %21 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }

    %2 = VPU.NCE.ClusterTiling (%1 as %arg0: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %filter as %arg1: tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable as %arg2: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !DistributedTileOutput {
        %21 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2, %arg3)
            {activation_window_channel_length = 18 : i64,
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
            post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"},
            rawFilterShape = [48, 1, 3, 3],
            strides = [1, 1]}
            -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }

    %3 = VPU.NCE.ClusterTiling (%2 as %arg0: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        %21 = VPU.Copy(%arg0) : tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        VPU.Yield %21
    }

    // Concat input tile 2
    %4 = VPU.Slice %input [0, 48, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>

    %5 = VPU.NCE.ClusterTiling (%4 as %arg0: tensor<1x48x32x32xf16, {order = #NHWC}>) -> !DistributedTile {
        %21 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }

    %6 = VPU.NCE.ClusterTiling (
        %5 as %arg0: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %filter as %arg1: tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable as %arg2: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !DistributedTileOutput {
        %21 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2, %arg3)
            {activation_window_channel_length = 18 : i64,
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
            post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"},
            rawFilterShape = [48, 1, 3, 3],
            strides = [1, 1]}
            -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }

    %7 = VPU.NCE.ClusterTiling (%6 as %arg0: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        %21 = VPU.Copy(%arg0) : tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        VPU.Yield %21
    }

    // Concat input tile 3
    %8 = VPU.Slice %input [0, 96, 0, 0] [1, 48, 32, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x48x32x32xf16, {order = #NHWC}>

    %9 = VPU.NCE.ClusterTiling (%8 as %arg0: tensor<1x48x32x32xf16, {order = #NHWC}>) -> !DistributedTile {
        %21 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }

    %10 = VPU.NCE.ClusterTiling (
        %9 as %arg0: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %filter as %arg1: tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable as %arg2: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !DistributedTileOutput {
        %21 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2, %arg3)
            {activation_window_channel_length = 18 : i64,
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
            post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64},
            name = "IE.Clamp"}, rawFilterShape = [48, 1, 3, 3],
            strides = [1, 1]}
            -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }

    %11 = VPU.NCE.ClusterTiling (%10 as %arg0: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        %21 = VPU.Copy(%arg0) : tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        VPU.Yield %21
    }

    %12 = VPU.Concat(%3, %7, %11) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]]} : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x144x32x32xf16, {order = #NHWC}>

    // Output branch 1
    %13 = VPU.Slice %12 [0, 0, 0, 0] [1, 144, 16, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x144x16x32xf16, {order = #NHWC}>
    %14 = VPU.NCE.ClusterTiling (%13 as %arg0: tensor<1x144x16x32xf16, {order = #NHWC}>) -> !Distributed {
        %21 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x144x16x32xf16, {order = #NHWC}> -> tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }

    %15 = VPU.NCE.ClusterTiling (
        %14 as %arg0: tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %maxPoolWeightsTable as %arg1: tensor<144x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow as %arg2: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !Distributed {
        %21 = VPU.NCE.MaxPool(%arg0, %arg1, %arg2) {
                activation_window_channel_length = 4 : i64,
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }

    %16 = VPU.NCE.ClusterTiling (%15 as %arg0: tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x144x16x32xf16, {order = #NHWC}> {
        %21 = VPU.Copy(%arg0) : tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x144x16x32xf16, {order = #NHWC}>
        VPU.Yield %21
    }

    // Output branch 2
    %17 = VPU.Slice %12 [0, 0, 16, 0] [1, 144, 16, 32] : tensor<1x144x32x32xf16, {order = #NHWC}> to tensor<1x144x16x32xf16, {order = #NHWC}>
    %18 = VPU.NCE.ClusterTiling (%17 as %arg0: tensor<1x144x16x32xf16, {order = #NHWC}>) -> !Distributed {
        %21 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x144x16x32xf16, {order = #NHWC}> -> tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }

    %19 = VPU.NCE.ClusterTiling (
        %18 as %arg0: tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %maxPoolWeightsTable as %arg1: tensor<144x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow as %arg2: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !Distributed {
        %21 = VPU.NCE.MaxPool(%arg0, %arg1, %arg2) {
                activation_window_channel_length = 4 : i64,
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                strides = [1, 1],
                kernel_size = [1, 1]
            } -> tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %21
    }

    %20 = VPU.NCE.ClusterTiling (%19 as %arg0: tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x144x16x32xf16, {order = #NHWC}> {
        %21 = VPU.Copy(%arg0) : tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x144x16x32xf16, {order = #NHWC}>
        VPU.Yield %21
    }

    return %16, %20 : tensor<1x144x16x32xf16, {order = #NHWC}>, tensor<1x144x16x32xf16, {order = #NHWC}>

}
}

// CHECK-LABEL: @CaseWithNceClusterTilingWithChildTiling

// Tile 0
// CHECK:       [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 48, 32, 32]

// CHECK:       [[COPY0:%.+]]  = VPU.NCE.ClusterTiling ([[SLICE0]] as %arg5: tensor<1x48x32x32xf16, {order = #NHWC}>)
// CHECK-SAME:      -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
// CHECK:           [[COPY0_INNER:%.+]] = VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x48x32x32xf16, {order = #NHWC}>
// CHECK-SAME:              -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[NCE0:%.+]]   = VPU.NCE.ClusterTiling ([[COPY0]] as %arg5: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
// CHECK:           [[NCE0_INNER:%.+]] = VPU.NCE.DepthConvolution(%arg5, %arg6, %arg7, %arg8)
// CHECK-SAME:              -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// Tile 1
// CHECK:       [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 48, 0, 0] [1, 48, 32, 32]

// CHECK:       [[COPY1:%.+]]  = VPU.NCE.ClusterTiling ([[SLICE1]] as %arg5: tensor<1x48x32x32xf16, {order = #NHWC}>)
// CHECK-SAME:      -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
// CHECK:           [[COPY1_INNER:%.+]] = VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x48x32x32xf16, {order = #NHWC}>
// CHECK-SAME:              -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[NCE1:%.+]]   = VPU.NCE.ClusterTiling ([[COPY1]] as %arg5: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
// CHECK:           [[NCE1_INNER:%.+]] = VPU.NCE.DepthConvolution(%arg5, %arg6, %arg7, %arg8)
// CHECK-SAME:              -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// Tile 2
// CHECK:       [[SLICE2:%.+]] = VPU.Slice %arg0 [0, 96, 0, 0] [1, 48, 32, 32]

// CHECK:       [[COPY2:%.+]]  = VPU.NCE.ClusterTiling ([[SLICE2]] as %arg5: tensor<1x48x32x32xf16, {order = #NHWC}>)
// CHECK-SAME:      -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
// CHECK:           [[COPY2_INNER:%.+]] = VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x48x32x32xf16, {order = #NHWC}>
// CHECK-SAME:              -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[NCE2:%.+]]   = VPU.NCE.ClusterTiling ([[COPY2]] as %arg5: tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
// CHECK:           [[NCE1_INNER:%.+]] = VPU.NCE.DepthConvolution(%arg5, %arg6, %arg7, %arg8)
// CHECK-SAME:              -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// Concat is CMX-ed
// CHECK:       [[CONCAT:%.+]] = VPU.Concat([[NCE0]], [[NCE1]], [[NCE2]])
// CHECK-SAME:      !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>,
// CHECK-SAME:          -> !VPU.DistributedTensor<1x144x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

// DistributedCast to cast the distribution mode
// CHECK:       [[DIST_CAST:%.+]] = VPU.DistributedCast([[CONCAT]]
// CHECK-SAME:      !VPU.DistributedTensor<1x144x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>)
// CHECK-SAME:          -> !VPU.DistributedTensor<1x144x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

// Output Tile 0
// CHECK:       [[OUTSLICE0:%.+]] = VPU.Slice [[DIST_CAST]] [0, 0, 0, 0] [1, 144, 16, 32]

// CHECK:       [[MAXPOOL0:%.+]] = VPU.NCE.ClusterTiling
// CHECK-SAME:      ([[OUTSLICE0]] as %arg5: tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:           [[MAXPOOL0_INNER:%.+]] = VPU.NCE.MaxPool(%arg5, %arg6, %arg7)

// CHECK:       [[OUTCOPY0_OUT:%.+]] = VPU.NCE.ClusterTiling
// CHECK-SAME:      ([[MAXPOOL0]] as %arg5: tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK:           [[OUTCOPY0_OUT_INNER:%.+]] = VPU.Copy(%arg5)

// Output Tile 1
// CHECK:       [[OUTSLICE1:%.+]] = VPU.Slice [[DIST_CAST]] [0, 0, 16, 0] [1, 144, 16, 32]

// CHECK:       [[MAXPOOL1:%.+]] = VPU.NCE.ClusterTiling
// CHECK-SAME:      ([[OUTSLICE1]] as %arg5: tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:           [[MAXPOOL1_INNER:%.+]] = VPU.NCE.MaxPool(%arg5, %arg6, %arg7)

// CHECK:       [[OUTCOPY1_OUT:%.+]] = VPU.NCE.ClusterTiling
// CHECK-SAME:      ([[MAXPOOL1]] as %arg5: tensor<1x144x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK:           [[OUTCOPY1_OUT_INNER:%.+]] = VPU.Copy(%arg5)

// CHECK:       return [[OUTCOPY0_OUT]], [[OUTCOPY1_OUT]]
// CHECK-SAME:      : tensor<1x144x16x32xf16, {order = #NHWC}>, tensor<1x144x16x32xf16, {order = #NHWC}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Distributed = type !VPU.DistributedTensor<
    1x48x16x32xf16, #NHWC, @CMX_NN, {
   mode = "DUPLICATED",
    num_clusters = 4
}>

!DistributedTile = type !VPU.DistributedTensor<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
   mode = "DUPLICATED",
    num_clusters = 4
}>

!DistributedTileOutput = type !VPU.DistributedTensor<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
   mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

module @CaseWithEltwiseNceClusterTilingWithChildTiling {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16, {order = #NHWC}>
        DataInfo "filter" : tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
        DataInfo "weightsTable" : tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
        DataInfo "maxPoolWeightsTable" : tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
        DataInfo "activationWindow" : tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    }
    outputsInfo : {
        DataInfo "prob1" : tensor<1x48x16x32xf16, {order = #NHWC}>
    }

func @main(%input: tensor<1x48x32x32xf16, {order = #NHWC}>,
           %filter: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
           %weightsTable: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
           %maxPoolWeightsTable: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
           %activationWindow: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
           -> tensor<1x48x16x32xf16, {order = #NHWC}> {

    // Create a concat subgraph with three input tiles and two output users

    // Concat input tile 1
    %0 = VPU.Slice %input [0, 0, 0, 0] [1, 16, 32, 32] : tensor<1x48x32x32xf16, {order = #NHWC}> to tensor<1x16x32x32xf16, {order = #NHWC}>

    %1 = VPU.NCE.ClusterTiling (%0 as %arg0: tensor<1x16x32x32xf16, {order = #NHWC}>) -> !DistributedTile {
        %19 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %19
    }

    %2 = VPU.NCE.ClusterTiling (%1 as %arg0: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %filter as %arg1: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable as %arg2: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !DistributedTileOutput {
        %19 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2, %arg3)
            {activation_window_channel_length = 18 : i64,
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
            post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"},
            rawFilterShape = [16, 1, 3, 3],
            strides = [1, 1]}
            -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %19
    }

    %3 = VPU.NCE.ClusterTiling (%2 as %arg0: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x32x32xf16, {order = #NHWC}> {
        %19 = VPU.Copy(%arg0) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>
        VPU.Yield %19
    }

    // Concat input tile 2
    %4 = VPU.Slice %input [0, 16, 0, 0] [1, 16, 32, 32] : tensor<1x48x32x32xf16, {order = #NHWC}> to tensor<1x16x32x32xf16, {order = #NHWC}>

    %5 = VPU.NCE.ClusterTiling (%4 as %arg0: tensor<1x16x32x32xf16, {order = #NHWC}>) -> !DistributedTile {
        %19 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %19
    }

    %6 = VPU.NCE.ClusterTiling (
        %5 as %arg0: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %filter as %arg1: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable as %arg2: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !DistributedTileOutput {
        %19 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2, %arg3)
            {activation_window_channel_length = 18 : i64,
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
            post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"},
            rawFilterShape = [16, 1, 3, 3],
            strides = [1, 1]}
            -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %19
    }

    %7 = VPU.NCE.ClusterTiling (%6 as %arg0: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x32x32xf16, {order = #NHWC}> {
        %19 = VPU.Copy(%arg0) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>
        VPU.Yield %19
    }

    // Concat input tile 3
    %8 = VPU.Slice %input [0, 96, 0, 0] [1, 16, 32, 32] : tensor<1x48x32x32xf16, {order = #NHWC}> to tensor<1x16x32x32xf16, {order = #NHWC}>

    %9 = VPU.NCE.ClusterTiling (%8 as %arg0: tensor<1x16x32x32xf16, {order = #NHWC}>) -> !DistributedTile {
        %19 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %19
    }

    %10 = VPU.NCE.ClusterTiling (
        %9 as %arg0: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %filter as %arg1: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weightsTable as %arg2: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
        %activationWindow as %arg3: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
            -> !DistributedTileOutput {
        %19 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2, %arg3)
            {activation_window_channel_length = 18 : i64,
            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
            post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64},
            name = "IE.Clamp"}, rawFilterShape = [16, 1, 3, 3],
            strides = [1, 1]}
            -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %19
    }

    %11 = VPU.NCE.ClusterTiling (%10 as %arg0: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x32x32xf16, {order = #NHWC}> {
        %19 = VPU.Copy(%arg0) : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>
        VPU.Yield %19
    }

    %12 = VPU.Concat(%3, %7, %11) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 96, 0, 0]]} : tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>

    // Output branch 1
    %13 = VPU.Slice %12 [0, 0, 0, 0] [1, 48, 16, 32] : tensor<1x48x32x32xf16, {order = #NHWC}> to tensor<1x48x16x32xf16, {order = #NHWC}>
    %14 = VPU.NCE.ClusterTiling (%13 as %arg0: tensor<1x48x16x32xf16, {order = #NHWC}>) -> !Distributed {
        %19 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x48x16x32xf16, {order = #NHWC}> -> tensor<1x48x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %19
    }

    // Output branch 2
    %15 = VPU.Slice %12 [0, 0, 16, 0] [1, 48, 16, 32] : tensor<1x48x32x32xf16, {order = #NHWC}> to tensor<1x48x16x32xf16, {order = #NHWC}>
    %16 = VPU.NCE.ClusterTiling (%15 as %arg0: tensor<1x48x16x32xf16, {order = #NHWC}>) -> !Distributed {
        %19 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x48x16x32xf16, {order = #NHWC}> -> tensor<1x48x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %19
    }

    %17 = VPU.NCE.ClusterTiling (
        %14 as %arg0: tensor<1x48x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %16 as %arg1: tensor<1x48x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
            -> !Distributed {
        %19 = VPU.NCE.Eltwise(%arg0, %arg1) {
            op_type = "ADD"
        } -> tensor<1x48x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        VPU.Yield %19
    }
    %18 = VPU.NCE.ClusterTiling (%17 as %arg0: tensor<1x48x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x48x16x32xf16, {order = #NHWC}> {
        %19 = VPU.Copy(%arg0) : tensor<1x48x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x16x32xf16, {order = #NHWC}>
        VPU.Yield %19
    }
    return %18 : tensor<1x48x16x32xf16, {order = #NHWC}>

}
}

// CHECK-LABEL: @CaseWithEltwiseNceClusterTilingWithChildTiling

// Tile 0
// CHECK:       [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 32, 32]

// CHECK:       [[COPY0:%.+]]  = VPU.NCE.ClusterTiling ([[SLICE0]] as %arg5: tensor<1x16x32x32xf16, {order = #NHWC}>)
// CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
// CHECK:           [[COPY0_INNER:%.+]] = VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x16x32x32xf16, {order = #NHWC}>
// CHECK-SAME:              -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[NCE0:%.+]]   = VPU.NCE.ClusterTiling ([[COPY0]] as %arg5: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
// CHECK:           [[NCE0_INNER:%.+]] = VPU.NCE.DepthConvolution(%arg5, %arg6, %arg7, %arg8)
// CHECK-SAME:              -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// Tile 1
// CHECK:       [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 16, 0, 0] [1, 16, 32, 32]

// CHECK:       [[COPY1:%.+]]  = VPU.NCE.ClusterTiling ([[SLICE1]] as %arg5: tensor<1x16x32x32xf16, {order = #NHWC}>)
// CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
// CHECK:           [[COPY1_INNER:%.+]] = VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x16x32x32xf16, {order = #NHWC}>
// CHECK-SAME:              -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[NCE1:%.+]]   = VPU.NCE.ClusterTiling ([[COPY1]] as %arg5: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
// CHECK:           [[NCE1_INNER:%.+]] = VPU.NCE.DepthConvolution(%arg5, %arg6, %arg7, %arg8)
// CHECK-SAME:              -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// Tile 2
// CHECK:       [[SLICE2:%.+]] = VPU.Slice %arg0 [0, 96, 0, 0] [1, 16, 32, 32]

// CHECK:       [[COPY2:%.+]]  = VPU.NCE.ClusterTiling ([[SLICE2]] as %arg5: tensor<1x16x32x32xf16, {order = #NHWC}>)
// CHECK-SAME:      -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
// CHECK:           [[COPY2_INNER:%.+]] = VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x16x32x32xf16, {order = #NHWC}>
// CHECK-SAME:              -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[NCE2:%.+]]   = VPU.NCE.ClusterTiling ([[COPY2]] as %arg5: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
// CHECK:           [[NCE1_INNER:%.+]] = VPU.NCE.DepthConvolution(%arg5, %arg6, %arg7, %arg8)
// CHECK-SAME:              -> tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

// Concat is CMX-ed
// CHECK:       [[CONCAT:%.+]] = VPU.Concat([[NCE0]], [[NCE1]], [[NCE2]])
// CHECK-SAME:      !VPU.DistributedTensor<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>,
// CHECK-SAME:          -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

// DistributedCast to cast the distribution mode
// CHECK:       [[DIST_CAST:%.+]] = VPU.DistributedCast([[CONCAT]]
// CHECK-SAME:      !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>)
// CHECK-SAME:          -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

// Output Tile 0
// CHECK:       [[OUTSLICE0:%.+]] = VPU.Slice [[DIST_CAST]] [0, 0, 16, 0] [1, 48, 16, 32]
// Output Tile 1
// CHECK:       [[OUTSLICE1:%.+]] = VPU.Slice [[DIST_CAST]] [0, 0, 0, 0] [1, 48, 16, 32]

// Eltwise with two same inputs
// CHECK:       [[ELTWISE:%.+]] = VPU.NCE.ClusterTiling
// CHECK-SAME:      ([[OUTSLICE1]] as %arg5: tensor<1x48x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:      [[OUTSLICE0]] as %arg6: tensor<1x48x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK:           [[ELTWISE_INNER:%.+]] = VPU.NCE.Eltwise(%arg5, %arg6) {op_type = "ADD"}

// CHECK:       [[OUTCOPY_OUT:%.+]] = VPU.NCE.ClusterTiling
// CHECK-SAME:      ([[ELTWISE]] as %arg5: tensor<1x48x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK:           [[OUTCOPY_OUT_INNER:%.+]] = VPU.Copy(%arg5)

// CHECK:       return [[OUTCOPY_OUT]]
// CHECK-SAME:      : tensor<1x48x16x32xf16, {order = #NHWC}>

// RUN: vpux-opt --split-input-file --cmx-concat --canonicalize %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CaseWithoutChildTiling
module @CaseWithoutChildTiling attributes {VPU.arch = "VPUX30XX"} {
    
IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8, VPU.derateFactor = 6.000000e-01}
IE.MemoryResource 4194304 bytes of @CMX_UPA {VPU.bandwidth = 16, VPU.derateFactor = 8.500000e-01}
IE.MemoryResource 3548160 bytes of @CMX_NN {VPU.bandwidth = 32, VPU.derateFactor = 1.000000e+00}

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x144x64x64xf16, {order = #NHWC}>
        DataInfo "filter" : tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
        DataInfo "weightsTable" : tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
        DataInfo "activationWindow" : tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
        DataInfo "weightsTableMaxPool" : tensor<144x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x144x64x64xf16, {order = #NHWC}>
    }

func @main(%input: tensor<1x144x64x64xf16, {order = #NHWC}>,
           %filter: tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
           %weightsTable: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
           %activationWindow: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>,
           %weightsTableMaxPool: tensor<144x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
           -> tensor<1x144x64x64xf16, {order = #NHWC}> {

    // Create a concat subgraph with three input tiles and one output user
    
    // Concat input tile 1
    %0 = IE.Slice %input [0, 0, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    %1 = IE.Copy(%0) {out_mem_space = @CMX_NN} : tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %2 = VPU.NCE.DepthConvolution(%1, %filter, %weightsTable, %activationWindow) 
        {activation_window_channel_length = 18 : i64, 
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, 
        post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, 
        rawFilterShape = [48, 1, 3, 3], 
        strides = [1, 1]} 
        -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    // NCE copy-out to concatinate in DDR
    %3 = IE.Copy(%2) : tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x64x64xf16, {order = #NHWC}>
    
    // Concat input tile 2
    %4 = IE.Slice %input [0, 48, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    %5 = IE.Copy(%4) {out_mem_space = @CMX_NN} : tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    
    %6 = VPU.NCE.DepthConvolution(%5, %filter, %weightsTable, %activationWindow) 
        {activation_window_channel_length = 18 : i64, 
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, 
        post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, 
        rawFilterShape = [48, 1, 3, 3], 
        strides = [1, 1]} 
        -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    // NCE copy-out to concatinate in DDR
    %7 = IE.Copy(%6) : tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x64x64xf16, {order = #NHWC}>
    
    // Concat input tile 3
    %8 = IE.Slice %input [0, 96, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    %9 = IE.Copy(%8) {out_mem_space = @CMX_NN} : tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    
    %10 = VPU.NCE.DepthConvolution(%9, %filter, %weightsTable, %activationWindow) 
        {activation_window_channel_length = 18 : i64, 
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, 
        post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, 
        name = "IE.Clamp"}, rawFilterShape = [48, 1, 3, 3], 
        strides = [1, 1]} 
        -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    // NCE copy-out to concatinate in DDR
    %11 = IE.Copy(%10) : tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x64x64xf16, {order = #NHWC}>
    
    // Concat inputs are in DDR and Concat output is in DDR
    %12 = IE.Concat(%3, %7, %11) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]]} : tensor<1x48x64x64xf16, {order = #NHWC}>, tensor<1x48x64x64xf16, {order = #NHWC}>, tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x144x64x64xf16, {order = #NHWC}>
    
    // Concat result copy-in for NCE user
    %13 = IE.Copy(%12) {out_mem_space = @CMX_NN} : tensor<1x144x64x64xf16, {order = #NHWC}> -> tensor<1x144x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    
    %14 = VPU.NCE.MaxPool(%13, %weightsTableMaxPool, %activationWindow) {
            activation_window_channel_length = 4 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1], kernel_size = [1, 1]
        } -> tensor<1x144x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %15 = IE.Copy(%14) : tensor<1x144x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x144x64x64xf16, {order = #NHWC}>

    return %15 : tensor<1x144x64x64xf16, {order = #NHWC}>

    // the below checks that the concat is occuring in NNCMX and the copy operations
    // arround the concat are not used by the concat producer and consumer operations

    // input to the first tile copy-in (DDR->NNCMX) for activation and weights
    // CHECK:       [[VAL0:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      [0, 0, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = IE.Copy([[VAL0]])

    // first tile of NCE task
    // CHECK:       [[VAL2:%.+]] = VPU.NCE.DepthConvolution([[VAL1]], %arg1, %arg2, %arg3)

    // no copy-out to concatinate in DDR
    // CHECK-NOT:   IE.Copy

    // input to the second tile copy-in (DDR->NNCMX) for activation and weights
    // CHECK:       [[VAL3:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      [0, 48, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    // CHECK:       [[VAL4:%.+]] = IE.Copy([[VAL3]])

    // second tile of NCE task
    // CHECK:       [[VAL5:%.+]] = VPU.NCE.DepthConvolution([[VAL4]], %arg1, %arg2, %arg3)

    // no copy-out to concatinate in DDR
    // CHECK-NOT:   IE.Copy

    // input to the third tile copy-in (DDR->NNCMX) for activation and weights
    // CHECK:       [[VAL6:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      [0, 96, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    // CHECK:       [[VAL7:%.+]] = IE.Copy([[VAL6]])

    // third tile of NCE task
    // CHECK:       [[VAL8:%.+]] = VPU.NCE.DepthConvolution([[VAL7]], %arg1, %arg2, %arg3)

    // no copy-out to concatinate in DDR
    // CHECK-NOT:   IE.Copy

    // no copy-out (NNCMX->DDR) operations to concatinate in DDR
    // Concat in NNCMX using results of NCE tiles in NNCMX
    // CHECK:       [[VAL9:%.+]] = IE.Concat([[VAL2]], [[VAL5]], [[VAL8]])

    // no Concat buffer copy-in to NNCMX
    // CHECK-NOT:   IE.Copy

    // user of the concat uses result of concat without intermediate copy operation
    // CHECK:       [[VAL10:%.+]] = VPU.NCE.MaxPool([[VAL9]], %arg4, %arg3)
    // CHECK:       [[VAL11:%.+]] = IE.Copy([[VAL10]])

    // CHECK:       return [[VAL11:%.+]] : tensor<1x144x64x64xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CaseWithChildTiling
module @CaseWithChildTiling attributes {VPU.arch = "VPUX30XX"} {
    
IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8, VPU.derateFactor = 6.000000e-01}
IE.MemoryResource 4194304 bytes of @CMX_UPA {VPU.bandwidth = 16, VPU.derateFactor = 8.500000e-01}
IE.MemoryResource 3548160 bytes of @CMX_NN {VPU.bandwidth = 32, VPU.derateFactor = 1.000000e+00}

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x144x64x64xf16, {order = #NHWC}>
        DataInfo "filter" : tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
        DataInfo "weightsTable" : tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
        DataInfo "activationWindow" : tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
        DataInfo "filterCons" : tensor<144x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
        DataInfo "weightsTableCons" : tensor<144x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    }
    outputsInfo : {
        DataInfo "prob1" : tensor<1x144x32x64xf16, {order = #NHWC}>
        DataInfo "prob2" : tensor<1x144x32x64xf16, {order = #NHWC}>
    }

func @main(%input: tensor<1x144x64x64xf16, {order = #NHWC}>,
           %filter: tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
           %weightsTable: tensor<48x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
           %activationWindow: tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>,
           %filterCons: tensor<144x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
           %weightsTableCons: tensor<144x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
           -> (tensor<1x144x32x64xf16, {order = #NHWC}>, tensor<1x144x32x64xf16, {order = #NHWC}>) {

    // Create a concat subgraph with three input tiles and one output user
    
    // Concat input tile 1
    %0 = IE.Slice %input [0, 0, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    %1 = IE.Copy(%0) {out_mem_space = @CMX_NN} : tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %2 = VPU.NCE.DepthConvolution(%1, %filter, %weightsTable, %activationWindow) 
        {activation_window_channel_length = 18 : i64, 
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, 
        post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, 
        rawFilterShape = [48, 1, 3, 3], 
        strides = [1, 1]} 
        -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    // NCE copy-out to concatinate in DDR
    %3 = IE.Copy(%2) : tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x64x64xf16, {order = #NHWC}>
    
    // Concat input tile 2
    %4 = IE.Slice %input [0, 48, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    %5 = IE.Copy(%4) {out_mem_space = @CMX_NN} : tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    
    %6 = VPU.NCE.DepthConvolution(%5, %filter, %weightsTable, %activationWindow) 
        {activation_window_channel_length = 18 : i64, 
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, 
        post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, 
        rawFilterShape = [48, 1, 3, 3], 
        strides = [1, 1]} 
        -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    // NCE copy-out to concatinate in DDR
    %7 = IE.Copy(%6) : tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x64x64xf16, {order = #NHWC}>
    
    // Concat input tile 3
    %8 = IE.Slice %input [0, 96, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    %9 = IE.Copy(%8) {out_mem_space = @CMX_NN} : tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    
    %10 = VPU.NCE.DepthConvolution(%9, %filter, %weightsTable, %activationWindow) 
        {activation_window_channel_length = 18 : i64, 
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, 
        post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, 
        name = "IE.Clamp"}, rawFilterShape = [48, 1, 3, 3], 
        strides = [1, 1]} 
        -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    // NCE copy-out to concatinate in DDR
    %11 = IE.Copy(%10) : tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x64x64xf16, {order = #NHWC}>
    
    // Concat inputs are in DDR and Concat output is in DDR
    %12 = IE.Concat(%3, %7, %11) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]]} : tensor<1x48x64x64xf16, {order = #NHWC}>, tensor<1x48x64x64xf16, {order = #NHWC}>, tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x144x64x64xf16, {order = #NHWC}>
     
    %13 = IE.Slice %12 [0, 0, 0, 0] [1, 144, 32, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x144x32x64xf16, {order = #NHWC}>
    // Concat slice result copy-in for NCE user
    %14 = IE.Copy(%13) {out_mem_space = @CMX_NN} : tensor<1x144x32x64xf16, {order = #NHWC}> -> tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %15 = VPU.NCE.DepthConvolution(%14, %filterCons, %weightsTableCons, %activationWindow) {activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, rawFilterShape = [144, 1, 3, 3], strides = [1, 1]} -> tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    %16 = IE.Copy(%15) : tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x144x32x64xf16, {order = #NHWC}>
    
    %17 = IE.Slice %12 [0, 144, 0, 0] [1, 144, 32, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x144x32x64xf16, {order = #NHWC}>
    // Concat slice result copy-in for NCE user
    %18 = IE.Copy(%17) {out_mem_space = @CMX_NN} : tensor<1x144x32x64xf16, {order = #NHWC}> -> tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %19 = VPU.NCE.DepthConvolution(%18, %filterCons, %weightsTableCons, %activationWindow) {activation_window_channel_length = 18 : i64, pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, rawFilterShape = [144, 1, 3, 3], strides = [1, 1]} -> tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    %20 = IE.Copy(%19) : tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x144x32x64xf16, {order = #NHWC}>

    return %16, %20 : tensor<1x144x32x64xf16, {order = #NHWC}>, tensor<1x144x32x64xf16, {order = #NHWC}>

    // the below checks that the concat is occuring in NNCMX and the copy operations
    // arround the concat are not used by the concat producer and consumer operations

    // input to the first tile copy-in (DDR->NNCMX) for activation and weights
    // CHECK:       [[VAL0:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      [0, 0, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = IE.Copy([[VAL0]])

    // first tile of NCE task
    // CHECK:       [[VAL2:%.+]] = VPU.NCE.DepthConvolution([[VAL1]], %arg1, %arg2, %arg3)

    // no copy-out to concatinate in DDR
    // CHECK-NOT:   IE.Copy

    // input to the second tile copy-in (DDR->NNCMX) for activation and weights
    // CHECK:       [[VAL3:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      [0, 48, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    // CHECK:       [[VAL4:%.+]] = IE.Copy([[VAL3]])

    // second tile of NCE task
    // CHECK:       [[VAL5:%.+]] = VPU.NCE.DepthConvolution([[VAL4]], %arg1, %arg2, %arg3)

    // no copy-out to concatinate in DDR
    // CHECK-NOT:   IE.Copy

    // input to the third tile copy-in (DDR->NNCMX) for activation and weights
    // CHECK:       [[VAL6:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      [0, 96, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    // CHECK:       [[VAL7:%.+]] = IE.Copy([[VAL6]])

    // third tile of NCE task
    // CHECK:       [[VAL8:%.+]] = VPU.NCE.DepthConvolution([[VAL7]], %arg1, %arg2, %arg3)

    // no copy-out to concatinate in DDR
    // CHECK-NOT:   IE.Copy

    // no copy-out (NNCMX->DDR) operations to concatinate in DDR
    // Concat in NNCMX using results of NCE tiles in NNCMX
    // CHECK:       [[VAL9:%.+]] = IE.Concat([[VAL2]], [[VAL5]], [[VAL8]])

    // no Concat buffer copy-in to NNCMX
    // CHECK-NOT:   IE.Copy

    // users of concat which use part of the master buffer through slices
    // concat partial buffer user slicing output of concat in NNCMX
    // CHECK:       [[VAL10:%.+]] = IE.Slice [[VAL9]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 144, 32, 64] : tensor<1x144x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> to tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    
    // no Concat buffer slice copy-in to NNCMX
    // CHECK-NOT:   IE.Copy

    // concat user reading from NNCMX slice without intermediate copy operation
    // CHECK:       [[VAL11:%.+]] = VPU.NCE.DepthConvolution([[VAL10]], %arg4, %arg5, %arg3)
    // copy-out
    // CHECK:       [[VAL12:%.+]] = IE.Copy([[VAL11]])

    // user of second part of concat master buffer
    // CHECK:       [[VAL13:%.+]] = IE.Slice [[VAL9]]
    // CHECK-SAME:      [0, 144, 0, 0] [1, 144, 32, 64] : tensor<1x144x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> to tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    
    // no Concat buffer slice copy-in to NNCMX
    // CHECK-NOT:   IE.Copy

    // user reading from NNCMX without intermediate copy operation
    // CHECK:       [[VAL14:%.+]] = VPU.NCE.DepthConvolution([[VAL13]], %arg4, %arg5, %arg3)
    // copy-out
    // CHECK:       [[VAL15:%.+]] = IE.Copy([[VAL14]])

    // CHECK:       return [[VAL12:%.+]], [[VAL15:%.+]] : tensor<1x144x32x64xf16, {order = #NHWC}>, tensor<1x144x32x64xf16, {order = #NHWC}>
}

}

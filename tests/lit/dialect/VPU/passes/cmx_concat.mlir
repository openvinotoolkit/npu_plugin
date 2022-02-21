// RUN: vpux-opt --split-input-file --cmx-concat %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CaseWithoutChildTiling
module @CaseWithoutChildTiling attributes {VPU.arch = "KMB"} {
    
IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8, VPU.derateFactor = 6.000000e-01}
IE.MemoryResource 4194304 bytes of @CMX_UPA {VPU.bandwidth = 16, VPU.derateFactor = 8.500000e-01}
IE.MemoryResource 2361600 bytes of @CMX_NN {VPU.bandwidth = 32, VPU.derateFactor = 1.000000e+00}

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x144x64x64xf16, {order = #NHWC}>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x144x64x64xf16, {order = #NHWC}>
    }

func @main(%arg0: tensor<1x144x64x64xf16, {order = #NHWC}>) -> tensor<1x144x64x64xf16, {order = #NHWC}> {

    %cst_0 = const.Declare tensor<48x16x1x1xf16> = #const.Content<dense<2.0> : tensor<48x16x1x1xf16>>
    %cst_1 = const.Declare tensor<48x16x1x1xf16> = #const.Content<dense<2.0> : tensor<48x16x1x1xf16>>
    %cst_2 = const.Declare tensor<48x16x1x1xf16> = #const.Content<dense<2.0> : tensor<48x16x1x1xf16>>
    
    %2 = IE.Slice %arg0 [0, 0, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    %3 = IE.Copy(%2) {out_mem_space = @CMX_NN} : tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %4 = IE.Copy(%cst_0) {out_mem_space = @CMX_NN} : tensor<48x16x1x1xf16> -> tensor<48x16x1x1xf16, {mem_space = @CMX_NN}>
    %5 = VPU.NCE.DepthConvolution(%3, %4) (bias : #const.Content<dense<2.0> : tensor<1x144x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 48, 1, 1]>]>) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, rawFilterShape = [48, 1, 3, 3], strides = [1, 1]} -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    %6 = IE.Copy(%5) : tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x64x64xf16, {order = #NHWC}>
    
    %7 = IE.Slice %arg0 [0, 48, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    %8 = IE.Copy(%7) {out_mem_space = @CMX_NN} : tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %9 = IE.Copy(%cst_1) {out_mem_space = @CMX_NN} : tensor<48x16x1x1xf16> -> tensor<48x16x1x1xf16, {mem_space = @CMX_NN}>
    %10 = VPU.NCE.DepthConvolution(%8, %9) (bias : #const.Content<dense<2.0> : tensor<1x144x1x1xf16>, [#const.SubView<[0, 48, 0, 0], [1, 48, 1, 1]>]>) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, rawFilterShape = [48, 1, 3, 3], strides = [1, 1]} -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    %11 = IE.Copy(%10) : tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x64x64xf16, {order = #NHWC}>
    
    %12 = IE.Slice %arg0 [0, 96, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    %13 = IE.Copy(%12) {out_mem_space = @CMX_NN} : tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %14 = IE.Copy(%cst_2) {out_mem_space = @CMX_NN} : tensor<48x16x1x1xf16> -> tensor<48x16x1x1xf16, {mem_space = @CMX_NN}>
    %15 = VPU.NCE.DepthConvolution(%13, %14) (bias : #const.Content<dense<2.0> : tensor<1x144x1x1xf16>, [#const.SubView<[0, 96, 0, 0], [1, 48, 1, 1]>]>) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, rawFilterShape = [48, 1, 3, 3], strides = [1, 1]} -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    %16 = IE.Copy(%15) : tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x64x64xf16, {order = #NHWC}>
    
    %17 = IE.Concat(%6, %11, %16) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]]} : tensor<1x48x64x64xf16, {order = #NHWC}>, tensor<1x48x64x64xf16, {order = #NHWC}>, tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x144x64x64xf16, {order = #NHWC}>
    
    %18 = IE.Copy(%17) {out_mem_space = @CMX_NN} : tensor<1x144x64x64xf16, {order = #NHWC}> -> tensor<1x144x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    
    %19 = VPU.NCE.MaxPool(%18) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, strides = [1, 1], kernel_size = [1, 1]
        } : tensor<1x144x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x144x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %20 = IE.Copy(%19) : tensor<1x144x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x144x64x64xf16, {order = #NHWC}>

    return %20 : tensor<1x144x64x64xf16, {order = #NHWC}>

    // first tile inputs copy in
    // CHECK:       IE.Slice
    // CHECK-SAME:      [0, 0, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    // CHECK:       IE.Copy
    // CHECK:       IE.Copy

    // first tile
    // CHECK:       [[VAL0:%.+]] = VPU.NCE.DepthConvolution

    // second tile inputs copy in
    // CHECK:       IE.Slice
    // CHECK-SAME:      [0, 48, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    // CHECK:       IE.Copy
    // CHECK:       IE.Copy

    // second tile
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.DepthConvolution

    // third tile inputs copy in
    // CHECK:       IE.Slice
    // CHECK-SAME:      [0, 96, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    // CHECK:       IE.Copy
    // CHECK:       IE.Copy

    // third tile
    // CHECK:       [[VAL2:%.+]] = VPU.NCE.DepthConvolution

    // concat in CMX
    // CHECK:       [[VAL3:%.+]] = IE.Concat([[VAL0]], [[VAL1]], [[VAL2]])

    // CHECK:       VPU.NCE.MaxPool([[VAL3]])
    // CHECK:       IE.Copy
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CaseWithChildTiling
module @CaseWithChildTiling attributes {VPU.arch = "KMB"} {
    
IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8, VPU.derateFactor = 6.000000e-01}
IE.MemoryResource 4194304 bytes of @CMX_UPA {VPU.bandwidth = 16, VPU.derateFactor = 8.500000e-01}
IE.MemoryResource 3548160 bytes of @CMX_NN {VPU.bandwidth = 32, VPU.derateFactor = 1.000000e+00}

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x144x64x64xf16, {order = #NHWC}>
    }
    outputsInfo : {
        DataInfo "prob1" : tensor<1x144x32x64xf16, {order = #NHWC}>
        DataInfo "prob2" : tensor<1x144x32x64xf16, {order = #NHWC}>
    }

func @main(%arg0: tensor<1x144x64x64xf16, {order = #NHWC}>) -> 
                    (tensor<1x144x32x64xf16, {order = #NHWC}>, tensor<1x144x32x64xf16, {order = #NHWC}>) {

    %cst_0 = const.Declare tensor<48x16x1x1xf16> = #const.Content<dense<2.0> : tensor<48x16x1x1xf16>>
    %cst_1 = const.Declare tensor<48x16x1x1xf16> = #const.Content<dense<2.0> : tensor<48x16x1x1xf16>>
    %cst_2 = const.Declare tensor<48x16x1x1xf16> = #const.Content<dense<2.0> : tensor<48x16x1x1xf16>>
    %cst_3 = const.Declare tensor<72x16x1x1xf16> = #const.Content<dense<2.0> : tensor<72x16x1x1xf16>>
    %cst_4 = const.Declare tensor<72x16x1x1xf16> = #const.Content<dense<2.0> : tensor<72x16x1x1xf16>>
    
    %2 = IE.Slice %arg0 [0, 0, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    %3 = IE.Copy(%2) {out_mem_space = @CMX_NN} : tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %4 = IE.Copy(%cst_0) {out_mem_space = @CMX_NN} : tensor<48x16x1x1xf16> -> tensor<48x16x1x1xf16, {mem_space = @CMX_NN}>
    %5 = VPU.NCE.DepthConvolution(%3, %4) (bias : #const.Content<dense<2.0> : tensor<1x144x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 48, 1, 1]>]>) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, rawFilterShape = [48, 1, 3, 3], strides = [1, 1]} -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    %6 = IE.Copy(%5) : tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x64x64xf16, {order = #NHWC}>
    
    %7 = IE.Slice %arg0 [0, 48, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    %8 = IE.Copy(%7) {out_mem_space = @CMX_NN} : tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %9 = IE.Copy(%cst_1) {out_mem_space = @CMX_NN} : tensor<48x16x1x1xf16> -> tensor<48x16x1x1xf16, {mem_space = @CMX_NN}>
    %10 = VPU.NCE.DepthConvolution(%8, %9) (bias : #const.Content<dense<2.0> : tensor<1x144x1x1xf16>, [#const.SubView<[0, 48, 0, 0], [1, 48, 1, 1]>]>) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, rawFilterShape = [48, 1, 3, 3], strides = [1, 1]} -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    %11 = IE.Copy(%10) : tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x64x64xf16, {order = #NHWC}>
    
    %12 = IE.Slice %arg0 [0, 96, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    %13 = IE.Copy(%12) {out_mem_space = @CMX_NN} : tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %14 = IE.Copy(%cst_2) {out_mem_space = @CMX_NN} : tensor<48x16x1x1xf16> -> tensor<48x16x1x1xf16, {mem_space = @CMX_NN}>
    %15 = VPU.NCE.DepthConvolution(%13, %14) (bias : #const.Content<dense<2.0> : tensor<1x144x1x1xf16>, [#const.SubView<[0, 96, 0, 0], [1, 48, 1, 1]>]>) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, rawFilterShape = [48, 1, 3, 3], strides = [1, 1]} -> tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    %16 = IE.Copy(%15) : tensor<1x48x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x48x64x64xf16, {order = #NHWC}>
    
    %17 = IE.Concat(%6, %11, %16) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]]} : tensor<1x48x64x64xf16, {order = #NHWC}>, tensor<1x48x64x64xf16, {order = #NHWC}>, tensor<1x48x64x64xf16, {order = #NHWC}> -> tensor<1x144x64x64xf16, {order = #NHWC}>
        
    %18 = IE.Slice %17 [0, 0, 0, 0] [1, 144, 32, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x144x32x64xf16, {order = #NHWC}>
    %19 = IE.Copy(%18) {out_mem_space = @CMX_NN} : tensor<1x144x32x64xf16, {order = #NHWC}> -> tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %20 = IE.Copy(%cst_3) {out_mem_space = @CMX_NN} : tensor<72x16x1x1xf16> -> tensor<72x16x1x1xf16, {mem_space = @CMX_NN}>
    %21 = VPU.NCE.DepthConvolution(%19, %20) (bias : #const.Content<dense<2.0> : tensor<1x144x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 144, 1, 1]>]>) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, rawFilterShape = [144, 1, 3, 3], strides = [1, 1]} -> tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    %22 = IE.Copy(%21) : tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x144x32x64xf16, {order = #NHWC}>
    
    %23 = IE.Slice %17 [0, 144, 0, 0] [1, 144, 32, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x144x32x64xf16, {order = #NHWC}>
    %24 = IE.Copy(%23) {out_mem_space = @CMX_NN} : tensor<1x144x32x64xf16, {order = #NHWC}> -> tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %25 = IE.Copy(%cst_4) {out_mem_space = @CMX_NN} : tensor<72x16x1x1xf16> -> tensor<72x16x1x1xf16, {mem_space = @CMX_NN}>
    %26 = VPU.NCE.DepthConvolution(%24, %25) (bias : #const.Content<dense<2.0> : tensor<1x144x1x1xf16>, [#const.SubView<[0, 144, 0, 0], [1, 144, 1, 1]>]>) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, post_op = {attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}, name = "IE.Clamp"}, rawFilterShape = [144, 1, 3, 3], strides = [1, 1]} -> tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> 
    %27 = IE.Copy(%26) : tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x144x32x64xf16, {order = #NHWC}>

    return %22, %27 : tensor<1x144x32x64xf16, {order = #NHWC}>, tensor<1x144x32x64xf16, {order = #NHWC}>

    // first tile inputs copy in
    // CHECK:       IE.Slice
    // CHECK-SAME:      [0, 0, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    // CHECK:       IE.Copy
    // CHECK:       IE.Copy

    // first tile
    // CHECK:       [[VAL0:%.+]] = VPU.NCE.DepthConvolution

    // second tile inputs copy in
    // CHECK:       IE.Slice
    // CHECK-SAME:      [0, 48, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    // CHECK:       IE.Copy
    // CHECK:       IE.Copy

    // second tile
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.DepthConvolution

    // third tile inputs copy in
    // CHECK:       IE.Slice
    // CHECK-SAME:      [0, 96, 0, 0] [1, 48, 64, 64] : tensor<1x144x64x64xf16, {order = #NHWC}> to tensor<1x48x64x64xf16, {order = #NHWC}>
    // CHECK:       IE.Copy
    // CHECK:       IE.Copy

    // third tile
    // CHECK:       [[VAL2:%.+]] = VPU.NCE.DepthConvolution

    // concat in CMX
    // CHECK:       [[VAL3:%.+]] = IE.Concat([[VAL0]], [[VAL1]], [[VAL2]])

    // child tiling
    // CHECK:       [[VAL4:%.+]] = IE.Slice
    // CHECK-SAME:      [0, 0, 0, 0] [1, 144, 32, 64] : tensor<1x144x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> to tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL5:%.+]] = IE.Copy

    // child tile 1
    // CHECK:       [[VAL6:%.+]] = VPU.NCE.DepthConvolution([[VAL4]], [[VAL5]])
    // CHECK:       IE.Copy

    // CHECK:       [[VAL7:%.+]] = IE.Slice
    // CHECK-SAME:      [0, 144, 0, 0] [1, 144, 32, 64] : tensor<1x144x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}> to tensor<1x144x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL8:%.+]] = IE.Copy

    // child tile 2
    // CHECK:       [[VAL9:%.+]] = VPU.NCE.DepthConvolution([[VAL7]], [[VAL8]])
    // CHECK:       IE.Copy
}

}

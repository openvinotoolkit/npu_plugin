// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --correct-NCE-workloads %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConv
func @DepthConv(%arg0: tensor<1x96x40x40xf16, {order = #NHWC}>) -> tensor<1x96x40x40xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<96x1x4x4xf16, {order = #NHWC}> =
        #const.Content<dense<1.000000e+00> : tensor<96x1x4x4xf16>, [#const.Reorder<#NHWC>]>
    %wt = const.Declare tensor<96x1x1x4xsi32, {order = #NHWC}> =
        #const.Content<dense<10> : tensor<96x1x1x4xsi32>, [#const.Reorder<#NHWC>]>
    %aw = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> =
        #const.Content<dense<1> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]>

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x96x40x40xf16, {order = #NHWC}>
        -> tensor<1x96x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<96x1x4x4xf16, {order = #NHWC}>
        -> tensor<96x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32, {order = #NHWC}>
        -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.Copy(%aw) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8, {order = #NHWC}>
        -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1, %2, %3) {
            activation_window_channel_length = 28 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [96, 1, 4, 4],
            strides = [1, 1]
        } -> tensor<1x96x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload [0, 0, 0, 0] [1, 96, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
            }

    %5 = VPU.Copy(%4) : tensor<1x96x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x96x40x40xf16, {order = #NHWC}>

    return %5 : tensor<1x96x40x40xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<96x1x4x4xf16, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<96x1x1x4xsi32, {order = #NHWC}>
    // CHECK:       [[CST1:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x96x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<96x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL3:%.+]] = VPU.Copy([[CST1]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]], [[VAL2]], [[VAL3]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x96x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               VPU.DPU.Workload [0, 0, 0, 0] [1, 64, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    // CHECK:               VPU.DPU.Workload [0, 64, 0, 0] [1, 32, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    // CHECK:           }

    // CHECK:       [[VAL5:%.+]] = VPU.Copy([[VAL4]])
    // CHECK-SAME:      -> tensor<1x96x40x40xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]] : tensor<1x96x40x40xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvLarge
func @DepthConvLarge(%arg0: tensor<1x512x40x40xf16, {order = #NHWC}>) -> tensor<1x512x40x40xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<512x1x4x4xf16, {order = #NHWC}> =
        #const.Content<dense<1.000000e+00> : tensor<512x1x4x4xf16>, [#const.Reorder<#NHWC>]>
    %wt = const.Declare tensor<512x1x1x4xsi32, {order = #NHWC}> =
        #const.Content<dense<10> : tensor<512x1x1x4xsi32>, [#const.Reorder<#NHWC>]>
    %aw = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> =
        #const.Content<dense<1> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]>

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x512x40x40xf16, {order = #NHWC}>
        -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<512x1x4x4xf16, {order = #NHWC}>
        -> tensor<512x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<512x1x1x4xsi32, {order = #NHWC}>
        -> tensor<512x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.Copy(%aw) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8, {order = #NHWC}>
        -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1, %2, %3) {
            activation_window_channel_length = 28 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [512, 1, 4, 4],
            strides = [1, 1]
        } -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload [0, 0, 0, 0] [1, 496, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
                VPU.DPU.Workload [0, 496, 0, 0] [1, 16, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
            }

    %5 = VPU.Copy(%4) : tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x512x40x40xf16, {order = #NHWC}>

    return %5 : tensor<1x512x40x40xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<512x1x4x4xf16, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<512x1x1x4xsi32, {order = #NHWC}>
    // CHECK:       [[CST1:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<512x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<512x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL3:%.+]] = VPU.Copy([[CST1]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]], [[VAL2]], [[VAL3]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    //CHECK:                VPU.DPU.Workload [0, 0, 0, 0] [1, 64, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    //CHECK:                VPU.DPU.Workload [0, 64, 0, 0] [1, 64, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    //CHECK:                VPU.DPU.Workload [0, 128, 0, 0] [1, 64, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    //CHECK:                VPU.DPU.Workload [0, 192, 0, 0] [1, 64, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    //CHECK:                VPU.DPU.Workload [0, 256, 0, 0] [1, 64, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    //CHECK:                VPU.DPU.Workload [0, 320, 0, 0] [1, 64, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    //CHECK:                VPU.DPU.Workload [0, 384, 0, 0] [1, 64, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    //CHECK:                VPU.DPU.Workload [0, 448, 0, 0] [1, 32, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    //CHECK:                VPU.DPU.Workload [0, 480, 0, 0] [1, 16, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    //CHECK:                VPU.DPU.Workload [0, 496, 0, 0] [1, 16, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    // CHECK:           }

    // CHECK:       [[VAL5:%.+]] = VPU.Copy([[VAL4]])
    // CHECK-SAME:      -> tensor<1x512x40x40xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]] : tensor<1x512x40x40xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvMultipleSplits
func @DepthConvMultipleSplits(%arg0: tensor<1x512x40x40xf16, {order = #NHWC}>) -> tensor<1x512x40x40xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<512x1x4x4xf16, {order = #NHWC}> =
        #const.Content<dense<1.000000e+00> : tensor<512x1x4x4xf16>, [#const.Reorder<#NHWC>]>
    %wt = const.Declare tensor<512x1x1x4xsi32, {order = #NHWC}> =
        #const.Content<dense<10> : tensor<512x1x1x4xsi32>, [#const.Reorder<#NHWC>]>
    %aw = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> =
        #const.Content<dense<1> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]>

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x512x40x40xf16, {order = #NHWC}>
        -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<512x1x4x4xf16, {order = #NHWC}>
        -> tensor<512x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<512x1x1x4xsi32, {order = #NHWC}>
        -> tensor<512x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.Copy(%aw) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8, {order = #NHWC}>
        -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1, %2, %3) {
            activation_window_channel_length = 28 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [512, 1, 4, 4],
            strides = [1, 1]
        } -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload [0, 0, 0, 0] [1, 80, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
                VPU.DPU.Workload [0, 80, 0, 0] [1, 16, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
                VPU.DPU.Workload [0, 96, 0, 0] [1, 96, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
            }

    %5 = VPU.Copy(%4) : tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x512x40x40xf16, {order = #NHWC}>

    return %5 : tensor<1x512x40x40xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<512x1x4x4xf16, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<512x1x1x4xsi32, {order = #NHWC}>
    // CHECK:       [[CST1:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<512x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<512x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL3:%.+]] = VPU.Copy([[CST1]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]], [[VAL2]], [[VAL3]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    //CHECK:                VPU.DPU.Workload [0, 0, 0, 0] [1, 64, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    //CHECK:                VPU.DPU.Workload [0, 64, 0, 0] [1, 16, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    //CHECK:                VPU.DPU.Workload [0, 80, 0, 0] [1, 16, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    //CHECK:                VPU.DPU.Workload [0, 96, 0, 0] [1, 64, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    //CHECK:                VPU.DPU.Workload [0, 160, 0, 0] [1, 32, 40, 80] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "CUBOID_4x16"
    // CHECK:           }

    // CHECK:       [[VAL5:%.+]] = VPU.Copy([[VAL4]])
    // CHECK-SAME:      -> tensor<1x512x40x40xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]] : tensor<1x512x40x40xf16, {order = #NHWC}>
}

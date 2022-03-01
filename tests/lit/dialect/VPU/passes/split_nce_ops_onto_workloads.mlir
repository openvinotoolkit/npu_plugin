// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB compilation-mode=DefaultHW" --split-NCE-ops-onto-workloads %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvRewriter
func @ConvRewriter(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        #const.Content<dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> =
        #const.Content<dense<10> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]>

    %0 = IE.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x16x16xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = IE.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<16x16x1x1xf16, {order = #NHWC}>
        -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = IE.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32, {order = #NHWC}>
        -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.NCE.Convolution(%0, %1, %2) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, 
            tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, 
            tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = IE.Copy(%3) : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %4 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = IE.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = IE.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = IE.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload [0, 0, 0, 0] [1, 16, 3, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 3, 0] [1, 16, 3, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 6, 0] [1, 16, 3, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 9, 0] [1, 16, 3, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 12, 0] [1, 16, 4, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = IE.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvRewriter
func @DepthConvRewriter(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> =
        #const.Content<dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]>
    %wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> =
        #const.Content<dense<10> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]>
    %aw = const.Declare tensor<16x1x1x16xui8, {order = #NHWC}> =
        #const.Content<dense<1> : tensor<16x1x1x16xui8>, [#const.Reorder<#NHWC>]>

    %0 = IE.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x40x80xf16, {order = #NHWC}>
        -> tensor<1x16x40x80xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = IE.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<16x1x4x8xf16, {order = #NHWC}>
        -> tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = IE.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32, {order = #NHWC}>
        -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = IE.Copy(%aw) {out_mem_space = @CMX_NN} : tensor<16x1x1x16xui8, {order = #NHWC}>
        -> tensor<16x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1, %2, %3) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1]
        } : tensor<1x16x40x80xf16, {mem_space = @CMX_NN, order = #NHWC}>, 
            tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<16x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %5 = IE.Copy(%4) : tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %5 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>
    // CHECK:       [[CST1:%.+]] = const.Declare tensor<16x1x1x16xui8, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = IE.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x40x80xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = IE.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = IE.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL3:%.+]] = IE.Copy([[CST1]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]], [[VAL2]], [[VAL3]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload [0, 0, 0, 0] [1, 16, 7, 73] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 7, 0] [1, 16, 7, 73] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 14, 0] [1, 16, 7, 73] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 21, 0] [1, 16, 7, 73] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 28, 0] [1, 16, 9, 73] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:           }

    // CHECK:       [[VAL5:%.+]] = IE.Copy([[VAL4]])
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]] : tensor<1x16x37x73xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolRewriter
func @MaxPoolRewriter(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %wt = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> =
        #const.Content<dense<10> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]>
    %aw = const.Declare tensor<16x1x1x16xui8, {order = #NHWC}> =
        #const.Content<dense<1> : tensor<16x1x1x16xui8>, [#const.Reorder<#NHWC>]>
    
    %0 = IE.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x1x4xf16, {order = #NHWC}>
        -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = IE.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32, {order = #NHWC}>
        -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = IE.Copy(%aw) {out_mem_space = @CMX_NN} : tensor<16x1x1x16xui8, {order = #NHWC}>
        -> tensor<16x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    %3 = VPU.NCE.MaxPool(%0, %1, %2) {
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1],
            kernel_size = [1, 1]
        } : tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<16x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = IE.Copy(%3) : tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x1x4xf16, {order = #NHWC}>

    return %4 : tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<16x1x1x16xui8, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = IE.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = IE.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = IE.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.MaxPool([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload [0, 0, 0, 0] [1, 16, 1, 4] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = IE.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x16x1x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddRewriter
func @EltwiseAddRewriter(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) { op_type = "ADD" } :
        tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %3 = IE.Copy(%2) : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %3 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = IE.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Eltwise([[VAL0]], [[VAL1]]) {op_type = "ADD"}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload [0, 0, 0, 0] [1, 64, 5, 28] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 5, 0] [1, 64, 5, 28] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 10, 0] [1, 64, 5, 28] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 15, 0] [1, 64, 5, 28] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 20, 0] [1, 64, 8, 28] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:           }

    // CHECK:       [[VAL3:%.+]] = IE.Copy([[VAL2]])
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[VAL3]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

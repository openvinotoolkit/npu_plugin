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
            rawFilterShape = [16, 16, 1, 1],
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
    %aw = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> =
        #const.Content<dense<1> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]>

    %0 = IE.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x40x80xf16, {order = #NHWC}>
        -> tensor<1x16x40x80xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = IE.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<16x1x4x8xf16, {order = #NHWC}>
        -> tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = IE.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32, {order = #NHWC}>
        -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = IE.Copy(%aw) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8, {order = #NHWC}>
        -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1, %2, %3) {
            activation_window_channel_length = 44 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            rawFilterShape = [16, 1, 4, 8],
            strides = [1, 1]
        } -> tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %5 = IE.Copy(%4) : tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %5 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>
    // CHECK:       [[CST1:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = IE.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x40x80xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = IE.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x4x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = IE.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL3:%.+]] = IE.Copy([[CST1]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]], [[VAL2]], [[VAL3]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               DPU.Workload [0, 0, 0, 0] [1, 16, 37, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 0, 16] [1, 16, 37, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 0, 32] [1, 16, 37, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 0, 48] [1, 16, 37, 16] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 0, 64] [1, 16, 37, 9] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
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
    %aw = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}> =
        #const.Content<dense<1> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]>

    %0 = IE.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x16x1x4xf16, {order = #NHWC}>
        -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = IE.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<16x1x1x4xsi32, {order = #NHWC}>
        -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = IE.Copy(%aw) {out_mem_space = @CMX_NN} : tensor<1x1x1x16xui8, {order = #NHWC}>
        -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

    %3 = VPU.NCE.MaxPool(%0, %1, %2) {
            activation_window_channel_length = 4 : i64,
            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
            strides = [1, 1],
            kernel_size = [1, 1]
        } -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = IE.Copy(%3) : tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x1x4xf16, {order = #NHWC}>

    return %4 : tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<1x1x1x16xui8, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = IE.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = IE.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = IE.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x1x1x16xui8, {mem_space = @CMX_NN, order = #NHWC}>

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
    // CHECK:               DPU.Workload [0, 0, 0, 0] [1, 64, 6, 28] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 6, 0] [1, 64, 6, 28] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 12, 0] [1, 64, 6, 28] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 18, 0] [1, 64, 6, 28] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:               DPU.Workload [0, 0, 24, 0] [1, 64, 4, 28] {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64} "VECTOR_FP16"
    // CHECK:           }

    // CHECK:       [[VAL3:%.+]] = IE.Copy([[VAL2]])
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[VAL3]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = type !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = OVERLAPPED,
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = {bottom = 1, left = 1, right = 1, top = 1},
    strides = [1, 1],
    num_clusters = 4
}>

!WeightsDistributed = type !VPU.DistributedTensor<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4
}>

!WeightsTableDistributed = type !VPU.DistributedTensor<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4
}>

!OutputDistributed = type !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = SEGMENTED,
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Input_DDR = type tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!Weights_DDR = type tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTable_DDR = type tensor<64x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
!Output_DDR = type tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

!InputStub_CMX = type tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsStub_CMX = type tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableStub_CMX = type tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = type tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>


func @ConvolutionWithDistributedTensor(%arg0: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]>
    %wt = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = #const.Content<dense<10> : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>>

    %input_cmx = VPU.NCE.ClusterTiling(%arg0 as %arg1: !Input_DDR) -> !InputDistributed {
        %0 = IE.Copy(%arg1) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputStub_CMX
        VPU.Yield %0
    }

    %weights_cmx = VPU.NCE.ClusterTiling(%weights as %arg1: !Weights_DDR) -> !WeightsDistributed {
        %0 = IE.Copy(%arg1) { out_mem_space = @CMX_NN } : !Weights_DDR -> !WeightsStub_CMX
        VPU.Yield %0
    }

    %wt_cmx = VPU.NCE.ClusterTiling(%wt as %arg1: !WeightsTable_DDR) -> !WeightsTableDistributed {
        %0 = IE.Copy(%arg1) { out_mem_space = @CMX_NN } : !WeightsTable_DDR -> !WeightsTableStub_CMX
        VPU.Yield %0
    }

    %output_cmx = VPU.NCE.ClusterTiling (
              %input_cmx as %arg1: !InputStub_CMX,
              %weights_cmx as %arg2: !WeightsStub_CMX,
              %wt_cmx as %arg3: !WeightsTableStub_CMX)
              -> !OutputDistributed {
        // Generate different workloads due to different pads on each cluster
        %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
                rawFilterShape = [64, 32, 3, 3],
                strides = [1, 1]
            } -> !OutputStub_CMX
        VPU.Yield %0
    }
    %output = VPU.NCE.ClusterTiling(%output_cmx as %arg1: !OutputStub_CMX) -> !Output_DDR {
        %0 = IE.Copy(%arg1) { out_mem_space = @DDR } : !OutputStub_CMX -> !Output_DDR
        VPU.Yield %0
    }

    return %output: !Output_DDR

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]>
    //CHECK:        [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = OVERLAPPED, num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, strides = [1, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES0:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}> -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>) -> !VPU.DistributedTensor<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 4 : i64}> {
    //CHECK:            [[RES1:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}> -> tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_TABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<64x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>) -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = DUPLICATED, num_clusters = 4 : i64}> {
    //CHECK:            [[RES2:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}> -> tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_TABLE_CMX]] as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:             -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:                [[RES4:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    //CHECK-SAME:                            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    //CHECK-SAME:                            strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                    VPU.DPU.Workload [0, 0, 0, 0] [1, 16, 4, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64} "VECTOR_FP16" attributes {cluster_id = 0 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 16, 0, 0] [1, 16, 4, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64} "VECTOR_FP16" attributes {cluster_id = 0 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 32, 0, 0] [1, 16, 4, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64} "VECTOR_FP16" attributes {cluster_id = 0 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 48, 0, 0] [1, 16, 4, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64} "VECTOR_FP16" attributes {cluster_id = 0 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 0, 4, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 1 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 0, 5, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 1 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 0, 6, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 1 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 0, 7, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 1 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 0, 8, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 2 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 0, 9, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 2 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 0, 10, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 2 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 0, 11, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 2 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 0, 12, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 3 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 0, 13, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 3 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 0, 14, 0] [1, 64, 1, 16] {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 3 : i64}
    //CHECK:                    VPU.DPU.Workload [0, 0, 15, 0] [1, 64, 1, 16] {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64} "VECTOR_FP16" attributes {cluster_id = 3 : i64}
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = IE.Copy(%arg1) {out_mem_space = @DDR} : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    //CHECK:        }
}

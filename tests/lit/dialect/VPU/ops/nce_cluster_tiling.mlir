// RUN: vpux-opt --split-input-file %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @ParsePrintClusterTiling(%arg0: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]>
    %wt = const.Declare tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = #const.Content<dense<10> : tensor<32x1x1x4xsi32, {mem_space = @CMX_NN}>>
    %aw = const.Declare tensor<32x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}> = #const.Content<dense<1> : tensor<32x1x1x16xui8, {mem_space = @CMX_NN}>>

    %0 = VPU.NCE.ClusterTiling (
            %arg0 as %arg1: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights as %arg2: tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %wt as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
            %aw as %arg4: tensor<32x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
                -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) (activationWindow : %arg4 : ) (bias : #const.Content<dense<1.000000e+00> : tensor<1x64x1x1xf16>>) {
                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                strides = [1, 1],
                activation_window_channel_length = 44
            } : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, 
                tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
                -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
      VPU.Yield %1
    }

    return %0 : tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[CST:%.*]] = const.Declare tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:        [[CST0:%.*]] = const.Declare tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        [[CST1:%.*]] = const.Declare tensor<32x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:                   %arg0 as %arg1: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                   [[CST]] as %arg2: tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                   [[CST0]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:                   [[CST1]] as %arg4: tensor<32x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                   -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    //CHECK:                [[VAL1:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
    //CHECK-SAME:                            (activationWindow : %arg4 : )
    //CHECK-SAME:                            (bias : #const.Content<dense<1.000000e+00> : tensor<1x64x1x1xf16>>) {
    //CHECK-SAME:                            activation_window_channel_length = 44 : i64, 
    //CHECK-SAME:                            pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, 
    //CHECK-SAME:                            strides = [1, 1]
    //CHECK-SAME:                } -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                VPU.Yield [[VAL1]]
    //CHECK:            }

    //CHECK:    return [[VAL0]] : tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = {bottom = 1, left = 1, right = 1, top = 1},
    num_clusters = 4
}>

!WeightsDistributed = type !VPU.DistributedTensor<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsTableDistributed = type !VPU.DistributedTensor<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4
}>

!ActivationWindowDistributed = type !VPU.DistributedTensor<
    32x1x1x16xui8, #NCHW, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4
}>

!OutputDistributed = type !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Input_DDR = type tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!Weights_DDR = type tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTable_DDR = type tensor<32x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
!ActivationWindow_DDR = type tensor<32x1x1x16xui8, {mem_space = @DDR, order = #NCHW}>
!Output_DDR = type tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

!InputStub_CMX = type tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsStub_CMX = type tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableStub_CMX = type tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!ActivationWindowStub_CMX = type tensor<32x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = type tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

func @ParsePrintDistributedTensor(%arg0: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}> = #const.Content<dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]>
    %wt = const.Declare tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = #const.Content<dense<10> : tensor<32x1x1x4xsi32, {mem_space = @CMX_NN}>>
    %aw = const.Declare tensor<32x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}> = #const.Content<dense<1> : tensor<32x1x1x16xui8, {mem_space = @CMX_NN}>>

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

    %aw_cmx = VPU.NCE.ClusterTiling(%aw as %arg1: !ActivationWindow_DDR) -> !ActivationWindowDistributed {
        %0 = IE.Copy(%arg1) { out_mem_space = @CMX_NN } : !ActivationWindow_DDR -> !ActivationWindowStub_CMX
        VPU.Yield %0
    }

    %output_cmx = VPU.NCE.ClusterTiling (
              %input_cmx as %arg1: !InputStub_CMX,
              %weights_cmx as %arg2: !WeightsStub_CMX,
              %wt_cmx as %arg3: !WeightsTableStub_CMX,
              %aw_cmx as %arg4: !ActivationWindowStub_CMX)
              -> !OutputDistributed {
        %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) (activationWindow : %arg4 : ) (bias : #const.Content<dense<1.000000e+00> : tensor<1x64x1x1xf16>>) {
                  pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
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
    //CHECK:        [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:        [[ACTIVATION_WINDOW:%.*]] = const.Declare tensor<32x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = OVERLAPPED, num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, num_clusters = 4 : i64}> {
    //CHECK:            [[RES0:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}> -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>) -> !VPU.DistributedTensor<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 4 : i64}> {
    //CHECK:            [[RES1:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}> -> tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_TABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<32x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>) -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = DUPLICATED, num_clusters = 4 : i64}> {
    //CHECK:            [[RES2:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}> -> tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[ACTIVATION_WINDOW_CMX:%.*]] = VPU.NCE.ClusterTiling ([[ACTIVATION_WINDOW]] as %arg1: tensor<32x1x1x16xui8, {mem_space = @DDR, order = #NCHW}>) -> !VPU.DistributedTensor<32x1x1x16xui8, #NCHW, @CMX_NN, {mode = DUPLICATED, num_clusters = 4 : i64}> {
    //CHECK:            [[RES3:%.*]] = IE.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<32x1x1x16xui8, {mem_space = @DDR, order = #NCHW}> -> tensor<32x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES3]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_TABLE_CMX]] as %arg3: tensor<32x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>,
    //CHECK-SAME:             [[ACTIVATION_WINDOW_CMX]] as %arg4: tensor<32x1x1x16xui8, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:             -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:                [[RES4:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3) 
    //CHECK-SAME:                            (activationWindow : %arg4 : )
    //CHECK-SAME:                            (bias : #const.Content<dense<1.000000e+00> : tensor<1x64x1x1xf16>>) {
    //CHECK-SAME:                            pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    //CHECK-SAME:                            strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = IE.Copy(%arg1) {out_mem_space = @DDR} : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
}

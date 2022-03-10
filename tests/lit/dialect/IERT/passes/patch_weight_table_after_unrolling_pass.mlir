// RUN: vpux-opt --split-input-file --patch-weight-table %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ActivationWindDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x16xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

// CHECK-LABEL: @PatchWeightTableWithNCEClusterTilingSOH
func @PatchWeightTableWithNCEClusterTilingSOH() -> !WeightsTableDistributed {
    %parent_input = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %in_1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    %in_2 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    %parent_output = VPURT.DeclareBuffer "CMX_NN" <17408> -> !OutputDistributed
    %out_1 = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    %out_2 = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    %weight = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !WeightsDistributed
    %weights_1 = VPURT.DeclareBuffer "CMX_NN" [0] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights_2 = VPURT.DeclareBuffer "CMX_NN" [1] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    %weight_table = VPURT.DeclareBuffer "CMX_NN" [0, 1] <35328> -> !WeightsTableDistributed
    %weight_table_1 = VPURT.DeclareBuffer "CMX_NN" [0] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %weight_table_2 = VPURT.DeclareBuffer "CMX_NN" [1] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    %act_wind = VPURT.DeclareBuffer "CMX_NN" [0, 1] <35584> -> !ActivationWindDistributed
    %act_wind_1 = VPURT.DeclareBuffer "CMX_NN" [0] <35584> -> memref<16x1x1x16xui8, [@CMX_NN, 0]>
    %act_wind_2 = VPURT.DeclareBuffer "CMX_NN" [1] <35584> -> memref<16x1x1x16xui8, [@CMX_NN, 1]>

    %weights_cst = const.Declare memref<16x16x1x1xf16, #NHWC> = #const.Content<dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.RelocateWeightsTable<34816 : i64, 35584 : i64>]>
    %activation_wind_cst = const.Declare memref<16x1x1x16xsi32> = #const.Content<dense<1> : tensor<16x1x1x16xsi32>>

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%weights_cst : memref<16x16x1x1xf16, #NHWC>) outputs(%weight : !WeightsDistributed) -> !WeightsDistributed
    }

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%weights_table_cst : memref<16x1x1x4xsi32>) outputs(%weight_table : !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%activation_wind_cst : memref<16x1x1x16xsi32>) outputs(%act_wind : !ActivationWindDistributed) -> !ActivationWindDistributed
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %1 = VPUIP.NCEClusterTask {
                    activation_window_channel_length = 98 : i64,
                    kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = "DWCONV"
                }
                input(%in_1 : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
                weights(%weights_1 : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
                weight_table(%weight_table_1 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
                activation_window(%act_wind_1 : memref<16x1x1x16xui8, [@CMX_NN, 0]>)
                parent_input(%parent_input : !InputDistributed)
                parent_output(%parent_output : !OutputDistributed)
                outputs(%out_1 : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
                -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
                variants : {
                    DPUTask {
                        end = [31, 16, 31],
                        mpe_mode = "VECTOR_FP16",
                        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                        start = [0, 0, 0]}
        } PPE :  {
        }
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %1 = VPUIP.NCEClusterTask {
                    activation_window_channel_length = 98 : i64,
                    kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = "DWCONV"
                }
                input(%in_2 : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
                weights(%weights_2 : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
                weight_table(%weight_table_2 : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
                activation_window(%act_wind_2 : memref<16x1x1x16xui8, [@CMX_NN, 1]>)
                parent_input(%parent_input : !InputDistributed)
                parent_output(%parent_output : !OutputDistributed)
                outputs(%out_2 : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
                -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
                variants : {
                    DPUTask {
                        end = [31, 32, 31],
                        mpe_mode = "VECTOR_FP16",
                        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                        start = [0, 17, 0]}
        } PPE :  {
        }
    }

    return %weight_table : !WeightsTableDistributed

    // CHECK:       [[WEIGHTS_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <[[WEIGHTS_ADDR:[^>]+]]>
    // CHECK:       [[WEIGHTS_BUF_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <[[WEIGHTS_ADDR:[^>]+]]> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHTS_BUF_2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <[[WEIGHTS_ADDR:[^>]+]]> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    // CHECK:       [[WEIGHT_TABLE_BUF_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHT_TABLE_BUF_2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    // CHECK:       [[ACTIVATION_WIND_BUF_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <[[ACTWIND_ADDR:[^>]+]]> -> memref<16x1x1x16xui8, [@CMX_NN, 0]>
    // CHECK:       [[ACTIVATION_WIND_BUF_2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <[[ACTWIND_ADDR:[^>]+]]> -> memref<16x1x1x16xui8, [@CMX_NN, 1]>
    // CHECK:       [[WEIGHTS_TABLE_CONST:%.*]] = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, [[ACTWIND_ADDR:[^>]+]] : i64>]>
    // CHECK:       [[NNDMA_OP:.*]] = VPUIP.NNDMA {port = 0 : i64} inputs([[WEIGHTS_TABLE_CONST]] : memref<16x1x1x4xsi32>)
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed = type !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x32x32xf16, {
        order = #NHWC,
        strides = [32768, 1, 1024, 32]
    }, 
    @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x32x32x32xf16, #NHWC, @DDR>
!Weights_DDR = type memref<32x16x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<32x1x1x4xsi32, #NCHW, @DDR>

!InputStub_CMX = type memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<32x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<32x1x1x4xsi32, #NCHW, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithNCEClusterTilingSOK
func @PatchWeightTableWithNCEClusterTilingSOK() -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]> {
    %parent_input = VPURT.DeclareBuffer "CMX_NN" <0> -> !ParentInputDistributed
    %in_1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    %in_2 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
    %parent_output = VPURT.DeclareBuffer "CMX_NN" <33792> -> !ParentOutputDistributed
    %parent_output_compact = VPURT.DeclareBuffer "CMX_NN" [0] <33792> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>
    %out_1 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <33792> -> !OutputDistributed
    %out_2 = VPURT.DeclareBuffer "CMX_NN" [1, 1] <33792> -> !OutputDistributed

    %weights_1 = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights_2 = VPURT.DeclareBuffer "CMX_NN" [1] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    %weight_table_1 = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %weight_table_2 = VPURT.DeclareBuffer "CMX_NN" [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    %act_wind_1 = VPURT.DeclareBuffer "CMX_NN" [0] <33536> -> memref<16x1x1x16xui8, [@CMX_NN, 0]>
    %act_wind_2 = VPURT.DeclareBuffer "CMX_NN" [1] <33536> -> memref<16x1x1x16xui8, [@CMX_NN, 1]>

    %weights_cst_1 = const.Declare memref<16x16x1x1xf16, #NHWC> = #const.Content<dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>]>
    %weights_cst_2 = const.Declare memref<16x16x1x1xf16, #NHWC> = #const.Content<dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[16, 0, 0, 0], [16, 16, 1, 1]>]>
    %weights_table_cst_1 = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 4]>]>
    %weights_table_cst_2 = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 4]>]>
    %activation_wind_cst_1 = const.Declare memref<16x1x1x16xui8> = #const.Content<dense<1> : tensor<32x1x1x16xui8>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 16]>]>
    %activation_wind_cst_2 = const.Declare memref<16x1x1x16xui8> = #const.Content<dense<1> : tensor<32x1x1x16xui8>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 16]>]>

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%weights_cst_1 : memref<16x16x1x1xf16, #NHWC>) outputs(%weights_1 : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%weights_cst_2 : memref<16x16x1x1xf16, #NHWC>) outputs(%weights_2 : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    }
    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%weights_table_cst_1 : memref<16x1x1x4xsi32>) outputs(%weight_table_1 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%weights_table_cst_2 : memref<16x1x1x4xsi32>) outputs(%weight_table_2 : memref<16x1x1x4xsi32, [@CMX_NN, 1]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    }
    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%activation_wind_cst_1 : memref<16x1x1x16xui8>) outputs(%act_wind_1 : memref<16x1x1x16xui8, [@CMX_NN, 0]>) -> memref<16x1x1x16xui8, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%activation_wind_cst_2 : memref<16x1x1x16xui8>) outputs(%act_wind_2 : memref<16x1x1x16xui8, [@CMX_NN, 1]>) -> memref<16x1x1x16xui8, [@CMX_NN, 1]>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %1 = VPUIP.NCEClusterTask {
                    activation_window_channel_length = 98 : i64,
                    kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = "DWCONV"
            }
            input(%in_1 : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%weights_1 : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%weight_table_1 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            activation_window(%act_wind_1 : memref<16x1x1x16xui8, [@CMX_NN, 0]>)
            parent_input(%parent_input : !ParentInputDistributed)
            parent_output(%parent_output : !ParentOutputDistributed)
            outputs(%out_1 : !OutputDistributed)
            -> !OutputDistributed
            variants : {
                DPUTask {
                    end = [31, 31, 15],
                    mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    start = [0, 0, 0]
                }
        } PPE :  {
        }
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %1 = VPUIP.NCEClusterTask {
                    activation_window_channel_length = 98 : i64,
                    kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = "DWCONV"
            }
            input(%in_2 : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
            weights(%weights_2 : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            weight_table(%weight_table_2 : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
            activation_window(%act_wind_2 : memref<16x1x1x16xui8, [@CMX_NN, 1]>)
            parent_input(%parent_input : !ParentInputDistributed)
            parent_output(%parent_output : !ParentOutputDistributed)
            outputs(%out_1 : !OutputDistributed)
            -> !OutputDistributed
            variants : {
                DPUTask {
                    end = [31, 31, 31],
                    mpe_mode = "VECTOR_FP16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                    start = [0, 0, 16]
                }
        } PPE :  {
        }
    }

    return %parent_output_compact : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[WEIGHTS_BUF_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <[[WEIGHTS_ADDR:[^>]+]]> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHTS_BUF_2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <[[WEIGHTS_ADDR:[^>]+]]> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    // CHECK:       [[WEIGHT_TABLE_BUF_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHT_TABLE_BUF_2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    // CHECK:       [[ACTIVATION_WIND_BUF_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <[[ACTWIND_ADDR:[^>]+]]> -> memref<16x1x1x16xui8, [@CMX_NN, 0]>
    // CHECK:       [[ACTIVATION_WIND_BUF_2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <[[ACTWIND_ADDR:[^>]+]]> -> memref<16x1x1x16xui8, [@CMX_NN, 1]>
    // CHECK:       [[CONST_1:%.*]] = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 4]>, #const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, [[ACTWIND_ADDR]] : i64>]>
    // CHECK:       [[CONST_2:%.*]] = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 4]>, #const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, [[ACTWIND_ADDR]] : i64>]>
    // CHECK:       [[NNDMA_OP_1:.*]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CONST_1]] : memref<16x1x1x4xsi32>) outputs([[WEIGHT_TABLE_BUF_1]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[NNDMA_OP_2:.*]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CONST_2]] : memref<16x1x1x4xsi32>) outputs([[WEIGHT_TABLE_BUF_2]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
}

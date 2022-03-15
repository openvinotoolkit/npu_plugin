// RUN: vpux-opt --split-input-file --patch-weight-table %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PatchWeightTable
func @PatchWeightTable() ->  memref<1008x1x1x4xsi32, @CMX_NN> {
    %weight_table = VPURT.DeclareBuffer "CMX_NN" <540288> -> memref<1008x1x1x4xsi32, @CMX_NN>
    %weights = VPURT.DeclareBuffer "CMX_NN" <395136> -> memref<1008x64x1x1xf16, #NHWC, @CMX_NN>
    %act_wind = VPURT.DeclareBuffer "CMX_NN" <524160> -> memref<1008x1x1x16xui8, @CMX_NN>
    %weight_table_const = const.Declare memref<1008x1x1x4xsi32> = #const.Content<dense<1> : tensor<1008x1x1x4xsi32>>

    %in = VPURT.DeclareBuffer "CMX_NN" <0> -> memref<1x1008x14x14xf16, #NHWC, @CMX_NN>
    %out = VPURT.DeclareBuffer "CMX_NN" <556416> -> memref<1x1008x2x2xf16, #NHWC, @CMX_NN>


    %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%weight_table_const : memref<1008x1x1x4xsi32>) outputs(%weight_table : memref<1008x1x1x4xsi32, @CMX_NN>) -> memref<1008x1x1x4xsi32, @CMX_NN>

    %5 = VPUIP.NCEClusterTask {activation_window_channel_length = 98 : i64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [7, 7], kernel_strides = [7, 7], task_type = "DWCONV"} input(%in : memref<1x1008x14x14xf16, #NHWC, @CMX_NN>) weights(%weights : memref<1008x64x1x1xf16, #NHWC, @CMX_NN>) weight_table(%weight_table : memref<1008x1x1x4xsi32, @CMX_NN>) activation_window(%act_wind : memref<1008x1x1x16xui8, @CMX_NN>) parent_input(%in : memref<1x1008x14x14xf16, #NHWC, @CMX_NN>) parent_output(%out : memref<1x1008x2x2xf16, #NHWC, @CMX_NN>) outputs(%out : memref<1x1008x2x2xf16, #NHWC, @CMX_NN>) -> memref<1x1008x2x2xf16, #NHWC, @CMX_NN> variants :  {
        DPUTask {end = [1, 0, 1007], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    } PPE :  {
    }

    return %weight_table : memref<1008x1x1x4xsi32, @CMX_NN>

    // CHECK:       [[WEIGHT_TABLE_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <540288> -> memref<1008x1x1x4xsi32, @CMX_NN>
    // CHECK:       [[WEIGHTS_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[WEIGHTS_ADDR:[^>]+]]> -> memref<1008x64x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:       [[ACT_WIN_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[ACT_WIN_ADDR:[^>]+]]> -> memref<1008x1x1x16xui8, @CMX_NN>
    // CHECK:       [[CONST:%.*]] = const.Declare memref<1008x1x1x4xsi32> = #const.Content<dense<1> : tensor<1008x1x1x4xsi32>, [#const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, [[ACT_WIN_ADDR]] : i64, [0]>]>
    // CHECK:       [[NDMA_OP:.*]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CONST]] : memref<1008x1x1x4xsi32>) outputs([[WEIGHT_TABLE_BUF]] : memref<1008x1x1x4xsi32, @CMX_NN>) -> memref<1008x1x1x4xsi32, @CMX_NN>
    // CHECK:       [[NCE_CLUST_TASK_OP:.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  weights([[WEIGHTS_BUF]] : memref<1008x64x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE_BUF]] : memref<1008x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:  activation_window([[ACT_WIN_BUF]] : memref<1008x1x1x16xui8, @CMX_NN>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PatchWeightTableActWinOnly
func @PatchWeightTableActWinOnly() ->  memref<16x1x1x4xsi32, @CMX_NN> {
    %weight_table = VPURT.DeclareBuffer "CMX_NN" <1024> ->  memref<16x1x1x4xsi32, @CMX_NN>
    %act_wind = VPURT.DeclareBuffer "CMX_NN" <508992> -> memref<16x1x1x16xui8, @CMX_NN>
    %weight_table_const = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>

    %in = VPURT.DeclareBuffer "CMX_NN" <4096> -> memref<1x16x113x113xf16, #NHWC, @CMX_NN>
    %out = VPURT.DeclareBuffer "CMX_NN" <408640> -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>

    %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%weight_table_const : memref<16x1x1x4xsi32>) outputs(%weight_table : memref<16x1x1x4xsi32, @CMX_NN>) -> memref<16x1x1x4xsi32, @CMX_NN>
    %5 = VPUIP.NCEClusterTask {activation_window_channel_length = 27 : i64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = "MAXPOOL"} input(%in : memref<1x16x113x113xf16, #NHWC, @CMX_NN>) weight_table(%weight_table : memref<16x1x1x4xsi32, @CMX_NN>) activation_window(%act_wind : memref<16x1x1x16xui8, @CMX_NN>) parent_input(%in : memref<1x16x113x113xf16, #NHWC, @CMX_NN>) parent_output(%out : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) outputs(%out : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x16x56x56xf16, #NHWC, @CMX_NN> variants :  {
      DPUTask {end = [55, 10, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    } PPE :  {
    }

    return %weight_table : memref<16x1x1x4xsi32, @CMX_NN>

    // CHECK:       [[WEIGHT_TABLE_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <1024> -> memref<16x1x1x4xsi32, @CMX_NN>
    // CHECK:       [[ACT_WIN_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[ACT_WIN_ADDR:[^>]+]]> -> memref<16x1x1x16xui8, @CMX_NN>
    // CHECK:       [[CONST:%.*]] = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.RelocateWeightsTable<0 : i64, [[ACT_WIN_ADDR]] : i64, [0]>]>
    // CHECK:       [[NDMA_OP:.*]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CONST]] : memref<16x1x1x4xsi32>) outputs([[WEIGHT_TABLE_BUF]] : memref<16x1x1x4xsi32, @CMX_NN>) -> memref<16x1x1x4xsi32, @CMX_NN>
    // CHECK:       [[NCE_CLUST_TASK_OP:.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  activation_window([[ACT_WIN_BUF]] : memref<16x1x1x16xui8, @CMX_NN>)

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PatchWeightTableWeightsOnly
func @PatchWeightTableWeightsOnly() -> memref<256x1x1x1xsi32, @CMX_NN> {
    %weight_table = VPURT.DeclareBuffer "CMX_NN" <1024> -> memref<256x1x1x1xsi32, @CMX_NN>
    %weights = VPURT.DeclareBuffer "CMX_NN" <4096> -> memref<64x16x3x3xf16, #NHWC, @CMX_NN>
    %weight_table_const = const.Declare memref<256x1x1x1xsi32> = #const.Content<dense<1> : tensor<256x1x1x1xsi32>>

    %in = VPURT.DeclareBuffer "CMX_NN" <0> -> memref<1x16x59x227xf16, #NHWC, @CMX_NN>
    %out = VPURT.DeclareBuffer "CMX_NN" <428608> -> memref<1x64x29x113xf16, #NHWC, @CMX_NN>
    %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%weight_table_const : memref<256x1x1x1xsi32>) outputs(%weight_table : memref<256x1x1x1xsi32, @CMX_NN>) -> memref<256x1x1x1xsi32, @CMX_NN>
    %5 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = "CONV"} input(%in : memref<1x16x59x227xf16, #NHWC, @CMX_NN>) weights(%weights : memref<64x16x3x3xf16, #NHWC, @CMX_NN>) weight_table(%weight_table : memref<256x1x1x1xsi32, @CMX_NN>) parent_input(%in : memref<1x16x59x227xf16, #NHWC, @CMX_NN>) parent_output(%out : memref<1x64x29x113xf16, #NHWC, @CMX_NN>) outputs(%out : memref<1x64x29x113xf16, #NHWC, @CMX_NN>) -> memref<1x64x29x113xf16, #NHWC, @CMX_NN> variants : {
      DPUTask {end = [112, 4, 63], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    } PPE :  {
    }

    return %weight_table : memref<256x1x1x1xsi32, @CMX_NN>

    // CHECK:       [[WEIGHT_TABLE_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <1024> -> memref<256x1x1x1xsi32, @CMX_NN>
    // CHECK:       [[WEIGHTS_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[WEIGHTS_ADDR:[^>]+]]> -> memref<64x16x3x3xf16, #NHWC, @CMX_NN>
    // CHECK:       [[CONST:%.*]] = const.Declare memref<256x1x1x1xsi32> = #const.Content<dense<1> : tensor<256x1x1x1xsi32>, [#const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, 16777215 : i64, [0]>]>
    // CHECK:       [[NDMA_OP:.*]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CONST]] : memref<256x1x1x1xsi32>) outputs([[WEIGHT_TABLE_BUF]] : memref<256x1x1x1xsi32, @CMX_NN>) -> memref<256x1x1x1xsi32, @CMX_NN>
    // CHECK:       [[NCE_CLUST_TASK_OP:.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  weights([[WEIGHTS_BUF]] : memref<64x16x3x3xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE_BUF]] : memref<256x1x1x1xsi32, @CMX_NN>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PatchWeightTableWithSpill
func @PatchWeightTableWithSpill() ->  memref<1x1008x2x2xf16, #NHWC, @CMX_NN> {
    %weight_table_1 = VPURT.DeclareBuffer "CMX_NN" <540288> -> memref<1008x1x1x4xsi32, @CMX_NN>
    %weight_table_DDR = VPURT.DeclareBuffer "DDR" <0> -> memref<1008x1x1x4xsi32, @DDR>
    %weight_table_2 = VPURT.DeclareBuffer "CMX_NN" <256> -> memref<1008x1x1x4xsi32, @CMX_NN>
    %weights = VPURT.DeclareBuffer "CMX_NN" <395136> -> memref<1008x64x1x1xf16, #NHWC, @CMX_NN>
    %act_wind = VPURT.DeclareBuffer "CMX_NN" <524160> -> memref<1008x1x1x16xui8, @CMX_NN>
    %weight_table_const = const.Declare memref<1008x1x1x4xsi32> = #const.Content<dense<1> : tensor<1008x1x1x4xsi32>>

    %in = VPURT.DeclareBuffer "CMX_NN" <0> -> memref<1x1008x14x14xf16, #NHWC, @CMX_NN>
    %out = VPURT.DeclareBuffer "CMX_NN" <556416> -> memref<1x1008x2x2xf16, #NHWC, @CMX_NN>

    %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%weight_table_const : memref<1008x1x1x4xsi32>) outputs(%weight_table_1 : memref<1008x1x1x4xsi32, @CMX_NN>) -> memref<1008x1x1x4xsi32, @CMX_NN>
    %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%weight_table_1 : memref<1008x1x1x4xsi32, @CMX_NN>) outputs(%weight_table_DDR : memref<1008x1x1x4xsi32, @DDR>) -> memref<1008x1x1x4xsi32, @DDR>
    %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%weight_table_DDR : memref<1008x1x1x4xsi32, @DDR>) outputs(%weight_table_2 : memref<1008x1x1x4xsi32, @CMX_NN>) -> memref<1008x1x1x4xsi32, @CMX_NN>

    %5 = VPUIP.NCEClusterTask {activation_window_channel_length = 98 : i64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [7, 7], kernel_strides = [7, 7], task_type = "DWCONV"} input(%in : memref<1x1008x14x14xf16, #NHWC, @CMX_NN>) weights(%weights : memref<1008x64x1x1xf16, #NHWC, @CMX_NN>) weight_table(%weight_table_2 : memref<1008x1x1x4xsi32, @CMX_NN>) activation_window(%act_wind : memref<1008x1x1x16xui8, @CMX_NN>) parent_input(%in : memref<1x1008x14x14xf16, #NHWC, @CMX_NN>) parent_output(%out : memref<1x1008x2x2xf16, #NHWC, @CMX_NN>) outputs(%out : memref<1x1008x2x2xf16, #NHWC, @CMX_NN>) -> memref<1x1008x2x2xf16, #NHWC, @CMX_NN> variants :  {
        DPUTask {end = [1, 0, 1007], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    } PPE :  {
    }

    return %5 : memref<1x1008x2x2xf16, #NHWC, @CMX_NN>

    // CHECK:       [[WEIGHT_TABLE_BUF1:%.*]] = VPURT.DeclareBuffer "CMX_NN" <540288> -> memref<1008x1x1x4xsi32, @CMX_NN>
    // CHECK:       [[WEIGHT_TABLE_BUF_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1008x1x1x4xsi32, @DDR>
    // CHECK:       [[WEIGHT_TABLE_BUF2:%.*]] = VPURT.DeclareBuffer "CMX_NN" <256> -> memref<1008x1x1x4xsi32, @CMX_NN>
    // CHECK:       [[WEIGHTS_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[WEIGHTS_ADDR:[^>]+]]> -> memref<1008x64x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:       [[ACT_WIN_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[ACT_WIN_ADDR:[^>]+]]> -> memref<1008x1x1x16xui8, @CMX_NN>
    // CHECK:       [[CONST:%.*]] = const.Declare memref<1008x1x1x4xsi32> = #const.Content<dense<1> : tensor<1008x1x1x4xsi32>, [#const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, [[ACT_WIN_ADDR]] : i64, [0]>]>
    // CHECK:       [[NDMA_OP1:.*]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CONST]] : memref<1008x1x1x4xsi32>) outputs([[WEIGHT_TABLE_BUF1]] : memref<1008x1x1x4xsi32, @CMX_NN>) -> memref<1008x1x1x4xsi32, @CMX_NN>
    // CHECK:       [[NDMA_OP2:.*]] = VPUIP.NNDMA
    // CHECK:       [[NDMA_OP3:.*]] = VPUIP.NNDMA
    // CHECK:       [[NCE_CLUST_TASK_OP:.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  weights([[WEIGHTS_BUF]] : memref<1008x64x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE_BUF2]] : memref<1008x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:  activation_window([[ACT_WIN_BUF]] : memref<1008x1x1x16xui8, @CMX_NN>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = SEGMENTED,
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = {bottom = 1, left = 1, right = 1, top = 1},
    num_clusters = 4
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = SEGMENTED,
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Input_DDR = type memref<1x32x16x16xf16, #NHWC, @DDR>
!Weights_DDR = type memref<64x32x3x3xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<64x1x1x4xsi32, #NHWC, @DDR>
!Output_DDR = type memref<1x64x16x16xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x32x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<64x32x3x3xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<64x1x1x4xsi32, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x64x16x16xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithNCEClusterTiling
func @PatchWeightTableWithNCEClusterTiling(%arg0: !Input_DDR) -> !Output_DDR {

    %buf_in = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %buf_W = VPURT.DeclareBuffer "CMX_NN" <16384> -> !WeightsDistributed
    %buf_WT = VPURT.DeclareBuffer "CMX_NN" <86016> -> !WeightsTableDistributed
    %buf_out = VPURT.DeclareBuffer "CMX_NN" <53248> -> !OutputDistributed
    %buf_out_DDR = VPURT.DeclareBuffer "DDR" <0> -> !Output_DDR

    %cst_W = const.Declare !Weights_DDR = #const.Content<dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]>
    %cst_WT = const.Declare !WeightsTable_DDR = #const.Content<dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]>

    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %8 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: !Input_DDR) outputs(%buf_in as %arg2: !InputStub_CMX) -> !InputDistributed {
        %9 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : !Input_DDR) outputs(%arg2 : !InputStub_CMX) -> !InputStub_CMX
      }
    }

    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %8 = VPUIP.NCEClusterTiling inputs(%cst_W as %arg1: !Weights_DDR) outputs(%buf_W as %arg2: !WeightsStub_CMX) -> !WeightsDistributed {
        %9 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : !Weights_DDR) outputs(%arg2 : !WeightsStub_CMX) -> !WeightsStub_CMX
      }
    }

    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %8 = VPUIP.NCEClusterTiling inputs(%cst_WT as %arg1: !WeightsTable_DDR) outputs(%buf_WT as %arg2: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
        %9 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : !WeightsTable_DDR) outputs(%arg2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
      }
    }

    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %8 = VPUIP.NCEClusterTiling inputs(%buf_in as %arg1: !InputStub_CMX, %buf_W as %arg2: !WeightsStub_CMX, %buf_WT as %arg3: !WeightsTableStub_CMX) outputs(%buf_out as %arg4: !OutputStub_CMX) -> !OutputStub_CMX {
        %9 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"} input(%arg1 : !InputStub_CMX) weights(%arg2 : !WeightsStub_CMX) weight_table(%arg3 : !WeightsTableStub_CMX) parent_input(%arg1 : !InputStub_CMX) parent_output(%arg4 : !OutputStub_CMX) outputs(%arg4 : !OutputStub_CMX) -> !OutputStub_CMX variants :  {
          DPUTask {end = [31, 15, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
        } PPE :  {
        }
      }
    }

    VPURT.Task waits(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %8 = VPUIP.NCEClusterTiling inputs(%buf_out as %arg1: !OutputStub_CMX) outputs(%buf_out_DDR as %arg2: !Output_DDR) -> !Output_DDR {
        %9 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : !OutputStub_CMX) outputs(%arg2 : !Output_DDR) -> !Output_DDR
      }
    }
    return %buf_out_DDR : !Output_DDR

    // CHECK:       [[INPUT_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer
    // CHECK:       [[WEIGHT_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[WEIGHTS_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer
    // CHECK:       [[WEIGHT_TABLE_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <86016> -> !VPUIP.DistributedBuffer
    // CHECK:       [[OUTPUT_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <53248> -> !VPUIP.DistributedBuffer
    // CHECK:       [[OUTPUT_BUF_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x64x16x16xf16, #NHWC, @DDR>

    // CHECK:       [[CONST_W:%.*]] = const.Declare memref<64x32x3x3xf16, #NHWC, @DDR>
    // CHECK:       [[CONST_WT:%.*]] = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = #const.Content<dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>, #const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, 16777215 : i64, [0, 0, 0, 0]>]>

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs(%arg0
    // CHECK-SAME:          outputs([[INPUT_BUF]]
    // CHECK-NEXT:          VPUIP.NNDMA 

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[CONST_W]]
    // CHECK-SAME:          outputs([[WEIGHT_BUF]]
    // CHECK-NEXT:          VPUIP.NNDMA   

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[CONST_WT]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF]]
    // CHECK-NEXT:          VPUIP.NNDMA 

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[INPUT_BUF]] as %arg1: memref<1x32x16x16xf16, #NHWC, @CMX_NN>, [[WEIGHT_BUF]] as %arg2: memref<64x32x3x3xf16, #NHWC, @CMX_NN>, [[WEIGHT_TABLE_BUF]] as %arg3: memref<64x1x1x4xsi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs([[OUTPUT_BUF]] as %arg4: memref<1x64x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-NEXT:          VPUIP.NCEClusterTask 

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[OUTPUT_BUF]]
    // CHECK-SAME:          outputs([[OUTPUT_BUF_DDR]]
    // CHECK-NEXT:          VPUIP.NNDMA 

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = SEGMENTED,
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = {bottom = 1, left = 1, right = 1, top = 1},
    num_clusters = 4
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = SEGMENTED,
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Input_DDR = type memref<1x32x16x16xf16, #NHWC, @DDR>
!Weights_DDR = type memref<64x32x3x3xf16, #NHWC, @DDR>
!WeightsTable_DDR = type memref<64x1x1x4xsi32, #NHWC, @DDR>
!Output_DDR = type memref<1x64x16x16xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x32x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<64x32x3x3xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<64x1x1x4xsi32, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x64x16x16xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithNCEClusterTilingWithSpilling
func @PatchWeightTableWithNCEClusterTilingWithSpilling(%arg0: !Input_DDR) -> !Output_DDR {

    %buf_in = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %buf_W = VPURT.DeclareBuffer "CMX_NN" <16384> -> !WeightsDistributed
    %buf_WT_1 = VPURT.DeclareBuffer "CMX_NN" <86016> -> !WeightsTableDistributed
    %buf_WT_DDR = VPURT.DeclareBuffer "DDR" <0> -> !WeightsTable_DDR
    %buf_WT_2 = VPURT.DeclareBuffer "CMX_NN" <86016> -> !WeightsTableDistributed
    %buf_out = VPURT.DeclareBuffer "CMX_NN" <53248> -> !OutputDistributed
    %buf_out_DDR = VPURT.DeclareBuffer "DDR" <0> -> !Output_DDR

    %cst_W = const.Declare !Weights_DDR = #const.Content<dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]>
    %cst_WT = const.Declare !WeightsTable_DDR = #const.Content<dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]>

    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %8 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %9 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %10 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: !Input_DDR) outputs(%buf_in as %arg2: !InputStub_CMX) -> !InputDistributed {
        %11 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : !Input_DDR) outputs(%arg2 : !InputStub_CMX) -> !InputStub_CMX
      }
    }

    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %10 = VPUIP.NCEClusterTiling inputs(%cst_W as %arg1: !Weights_DDR) outputs(%buf_W as %arg2: !WeightsStub_CMX) -> !WeightsDistributed {
        %11 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : !Weights_DDR) outputs(%arg2 : !WeightsStub_CMX) -> !WeightsStub_CMX
      }
    }

    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %10 = VPUIP.NCEClusterTiling inputs(%cst_WT as %arg1: !WeightsTable_DDR) outputs(%buf_WT_1 as %arg2: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
        %11 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : !WeightsTable_DDR) outputs(%arg2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
      }
    }

    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %10 = VPUIP.NCEClusterTiling inputs(%buf_WT_1 as %arg1: !WeightsTableStub_CMX) outputs(%buf_WT_DDR as %arg2: !WeightsTable_DDR) -> !WeightsTable_DDR {
        %11 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : !WeightsTableStub_CMX) outputs(%arg2 : !WeightsTable_DDR) -> !WeightsTable_DDR
      }
    }

    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %10 = VPUIP.NCEClusterTiling inputs(%buf_WT_DDR as %arg1: !WeightsTable_DDR) outputs(%buf_WT_2 as %arg2: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
        %11 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : !WeightsTable_DDR) outputs(%arg2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
      }
    }

    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %10 = VPUIP.NCEClusterTiling inputs(%buf_in as %arg1: !InputStub_CMX, %buf_W as %arg2: !WeightsStub_CMX, %buf_WT_2 as %arg3: !WeightsTableStub_CMX) outputs(%buf_out as %arg4: !OutputStub_CMX) -> !OutputStub_CMX {
        %11 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"} input(%arg1 : !InputStub_CMX) weights(%arg2 : !WeightsStub_CMX) weight_table(%arg3 : !WeightsTableStub_CMX) parent_input(%arg1 : !InputStub_CMX) parent_output(%arg4 : !OutputStub_CMX) outputs(%arg4 : !OutputStub_CMX) -> !OutputStub_CMX variants :  {
          DPUTask {end = [31, 15, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
        } PPE :  {
        }
      }
    }

    VPURT.Task waits(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %10 = VPUIP.NCEClusterTiling inputs(%buf_out as %arg1: !OutputStub_CMX) outputs(%buf_out_DDR as %arg2: !Output_DDR) -> !Output_DDR {
        %11 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : !OutputStub_CMX) outputs(%arg2 : !Output_DDR) -> !Output_DDR
      }
    }
    return %buf_out_DDR : !Output_DDR

    // CHECK:       [[INPUT_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer
    // CHECK:       [[WEIGHT_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[WEIGHTS_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer
    // CHECK:       [[WEIGHT_TABLE_BUF_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" <86016> -> !VPUIP.DistributedBuffer
    // CHECK:       [[WEIGHT_TABLE_BUF_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<64x1x1x4xsi32, #NHWC, @DDR>
    // CHECK:       [[WEIGHT_TABLE_BUF_2:%.*]] = VPURT.DeclareBuffer "CMX_NN" <86016> -> !VPUIP.DistributedBuffer
    // CHECK:       [[OUTPUT_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <53248> -> !VPUIP.DistributedBuffer
    // CHECK:       [[OUTPUT_BUF_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x64x16x16xf16, #NHWC, @DDR>

    // CHECK:       [[CONST_W:%.*]] = const.Declare memref<64x32x3x3xf16, #NHWC, @DDR>
    // CHECK:       [[CONST_WT:%.*]] = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = #const.Content<dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>, #const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, 16777215 : i64, [0, 0, 0, 0]>]>

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs(%arg0
    // CHECK-SAME:          outputs([[INPUT_BUF]]
    // CHECK-NEXT:          VPUIP.NNDMA 

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[CONST_W]]
    // CHECK-SAME:          outputs([[WEIGHT_BUF]]
    // CHECK-NEXT:          VPUIP.NNDMA   

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[CONST_WT]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF_1]]
    // CHECK-NEXT:          VPUIP.NNDMA 

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[WEIGHT_TABLE_BUF_1]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF_DDR]]
    // CHECK-NEXT:          VPUIP.NNDMA 

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[WEIGHT_TABLE_BUF_DDR]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF_2]]
    // CHECK-NEXT:          VPUIP.NNDMA 

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[INPUT_BUF]] as %arg1: memref<1x32x16x16xf16, #NHWC, @CMX_NN>, [[WEIGHT_BUF]] as %arg2: memref<64x32x3x3xf16, #NHWC, @CMX_NN>, [[WEIGHT_TABLE_BUF_2]] as %arg3: memref<64x1x1x4xsi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs([[OUTPUT_BUF]] as %arg4: memref<1x64x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-NEXT:          VPUIP.NCEClusterTask 

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[OUTPUT_BUF]]
    // CHECK-SAME:          outputs([[OUTPUT_BUF_DDR]]
    // CHECK-NEXT:          VPUIP.NNDMA 

}

// -----

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

!WeightsTable_DDR = type memref<16x1x1x4xsi32>

!InputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithNCEClusterTilingWithSOHAndWeightsOnly
func @PatchWeightTableWithNCEClusterTilingWithSOHAndWeightsOnly() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer "CMX_NN" <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer "CMX_NN" <35328> -> !WeightsTableDistributed

    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%weights_table_cst as %arg0: !WeightsTable_DDR)
              outputs(%weights_table as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
            VPUIP.NNDMA inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
        }
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling
                inputs(%parent_input_cmx as %arg0: !InputStub_CMX,
                        %weights as %arg1: !WeightsStub_CMX,
                        %weights_table as %arg2: !WeightsTableStub_CMX)
                outputs(%parent_out_cmx as %arg4: !OutputStub_CMX)
                    -> !OutputStub_CMX {

              %1 = VPUIP.NCEClusterTask {
                        kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                        kernel_size = [1, 1],
                        kernel_strides = [1, 1],
                        task_type = "CONV"
                    }  input(%arg0 : !InputStub_CMX)
                        weights(%arg1 : !WeightsStub_CMX)
                        weight_table(%arg2 : !WeightsTableStub_CMX)
                        parent_input(%arg0 : !InputStub_CMX)
                        parent_output(%arg4 : !OutputStub_CMX)
                        outputs(%arg4 : !OutputStub_CMX)
                            -> !OutputStub_CMX variants :  {
                          DPUTask {
                              start = [0, 0, 0], end = [31, 16, 31],
                              mpe_mode = "VECTOR_FP16",
                              pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                              cluster_id = 0 : i64
                          }
                          DPUTask {
                              start = [0, 17, 0], end = [31, 32, 31],
                              mpe_mode = "VECTOR_FP16",
                              pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                              cluster_id = 1 : i64
                          }
                        } PPE :  {
                        }
        }
    }

    return %parent_out_cmx : !OutputDistributed

    // CHECK:       [[WEIGHTS_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[WEIGHTS_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHT_TABLE_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <35328> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    // CHECK:       [[CONST:%.*]] = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, 16777215 : i64, [0, 0]>]>
    
    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[CONST]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF]]
    // CHECK-NEXT:          VPUIP.NNDMA
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

!WeightsTable_DDR = type memref<32x1x1x4xsi32, #NCHW, @DDR>

!InputStub_CMX = type memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<32x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<32x1x1x4xsi32, #NCHW, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithNCEClusterTilingWithSOKAndWeightsOnly
// For SOK, we get an incorrect weights table that will be rewritten after UnrollClusterTilingPass pass
func @PatchWeightTableWithNCEClusterTilingWithSOKAndWeightsOnly() -> !ParentOutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !ParentInputDistributed
    %weights = VPURT.DeclareBuffer "CMX_NN" <32768> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer "CMX_NN" <33280> -> !WeightsTableDistributed
    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <33536> -> !ParentOutputDistributed

    %weights_table_cst = const.Declare memref<32x1x1x4xsi32> = #const.Content<dense<1> : tensor<32x1x1x4xsi32>>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_table_cst as %arg0: !WeightsTable_DDR)
                outputs(%weights_table as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
             VPUIP.NNDMA inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
         }
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling
                 inputs(%parent_input_cmx as %arg0: !InputStub_CMX,
                         %weights as %arg1: !WeightsStub_CMX,
                         %weights_table as %arg2: !WeightsTableStub_CMX)
                 outputs(%parent_out_cmx as %arg3: !OutputStub_CMX)
                     -> !OutputStub_CMX {

               %1 = VPUIP.NCEClusterTask {
                         kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                         kernel_size = [1, 1],
                         kernel_strides = [1, 1],
                         task_type = "CONV"
                     }  input(%arg0 : !InputStub_CMX)
                         weights(%arg1 : !WeightsStub_CMX)
                         weight_table(%arg2 : !WeightsTableStub_CMX)
                         parent_input(%arg0 : !InputStub_CMX)
                         parent_output(%arg3 : !OutputStub_CMX)
                         outputs(%arg3 : !OutputStub_CMX)
                             -> !OutputStub_CMX variants :  {
                            DPUTask {
                                start = [0, 0, 0], end = [31, 31, 15],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                start = [0, 0, 16], end = [31, 31, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 1 : i64
                            }
                         } PPE :  {
                         }
        }
    }

    return %parent_out_cmx: !ParentOutputDistributed

    // CHECK:       [[WEIGHTS_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[WEIGHTS_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHT_TABLE_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <33280> -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = SEGMENTED, num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[CONST:%.*]] = const.Declare memref<32x1x1x4xsi32> = #const.Content<dense<1> : tensor<32x1x1x4xsi32>, [#const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, 16777215 : i64, [0, 16]>]>

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[CONST]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF]]
    // CHECK-NEXT:          VPUIP.NNDMA
}

// -----

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

!ActWindDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x16xui8, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTable_DDR = type memref<16x1x1x4xsi32>

!InputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!ActWindStub_CMX = type memref<16x1x1x16xui8, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithNCEClusterTilingWithSOH
func @PatchWeightTableWithNCEClusterTilingWithSOH() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer "CMX_NN" <34816> -> !WeightsDistributed
    %actWind = VPURT.DeclareBuffer "CMX_NN" <35328> -> !ActWindDistributed
    %weights_table = VPURT.DeclareBuffer "CMX_NN" <35584> -> !WeightsTableDistributed

    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%weights_table_cst as %arg0: !WeightsTable_DDR)
              outputs(%weights_table as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
            VPUIP.NNDMA inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
        }
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling
                inputs(%parent_input_cmx as %arg0: !InputStub_CMX,
                        %weights as %arg1: !WeightsStub_CMX,
                        %weights_table as %arg2: !WeightsTableStub_CMX,
                        %actWind as %arg3: !ActWindStub_CMX)
                outputs(%parent_out_cmx as %arg4: !OutputStub_CMX)
                    -> !OutputStub_CMX {

              %1 = VPUIP.NCEClusterTask {
                        activation_window_channel_length = 98 : i64,
                        kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                        kernel_size = [1, 1],
                        kernel_strides = [1, 1],
                        task_type = "DWCONV"
                    }  input(%arg0 : !InputStub_CMX)
                        weights(%arg1 : !WeightsStub_CMX)
                        weight_table(%arg2 : !WeightsTableStub_CMX)
                        activation_window(%arg3 : !ActWindStub_CMX)
                        parent_input(%arg0 : !InputStub_CMX)
                        parent_output(%arg4 : !OutputStub_CMX)
                        outputs(%arg4 : !OutputStub_CMX)
                            -> !OutputStub_CMX variants :  {
                          DPUTask {
                              start = [0, 0, 0], end = [31, 16, 31],
                              mpe_mode = "VECTOR_FP16",
                              pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                              cluster_id = 0 : i64
                          }
                          DPUTask {
                              start = [0, 17, 0], end = [31, 32, 31],
                              mpe_mode = "VECTOR_FP16",
                              pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                              cluster_id = 1 : i64
                          }
                        } PPE :  {
                        }
        }
    }

    return %parent_out_cmx : !OutputDistributed

    // CHECK:       [[WEIGHTS_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[WEIGHTS_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    // CHECK:       [[ACT_WIN_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[ACT_WIN_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer<16x1x1x16xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHT_TABLE_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <35584> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = DUPLICATED, num_clusters = 2 : i64}>
    // CHECK:       [[CONST:%.*]] = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, [[ACT_WIN_ADDR]] : i64, [0, 0]>]>
    
    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[CONST]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF]]
    // CHECK-NEXT:          VPUIP.NNDMA
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

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!ActWindDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x16xui8, #NCHW, @CMX_NN, {
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

!WeightsTable_DDR = type memref<32x1x1x4xsi32, #NCHW, @DDR>

!InputStub_CMX = type memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<32x16x1x1xf16, #NHWC, @CMX_NN>
!ActWindStub_CMX = type memref<16x1x1x16xui8, @CMX_NN>
!WeightsTableStub_CMX = type memref<32x1x1x4xsi32, #NCHW, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithNCEClusterTilingWithSOK
// For SOK, we get an incorrect weights table that will be rewritten after UnrollClusterTilingPass pass
func @PatchWeightTableWithNCEClusterTilingWithSOK() -> !ParentOutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !ParentInputDistributed
    %weights = VPURT.DeclareBuffer "CMX_NN" <32768> -> !WeightsDistributed
    %actWind = VPURT.DeclareBuffer "CMX_NN" <33536> -> !ActWindDistributed
    %weights_table = VPURT.DeclareBuffer "CMX_NN" <33280> -> !WeightsTableDistributed
    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <33664> -> !ParentOutputDistributed

    %weights_table_cst = const.Declare memref<32x1x1x4xsi32> = #const.Content<dense<1> : tensor<32x1x1x4xsi32>>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_table_cst as %arg0: !WeightsTable_DDR)
                outputs(%weights_table as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
             VPUIP.NNDMA inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
         }
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling
                 inputs(%parent_input_cmx as %arg0: !InputStub_CMX,
                         %weights as %arg1: !WeightsStub_CMX,
                         %weights_table as %arg2: !WeightsTableStub_CMX,
                         %actWind as %arg3: !ActWindStub_CMX)
                 outputs(%parent_out_cmx as %arg4: !OutputStub_CMX)
                     -> !OutputStub_CMX {

               %1 = VPUIP.NCEClusterTask {
                         activation_window_channel_length = 98 : i64,
                         kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                         kernel_size = [1, 1],
                         kernel_strides = [1, 1],
                         task_type = "DWCONV"
                     }  input(%arg0 : !InputStub_CMX)
                         weights(%arg1 : !WeightsStub_CMX)
                         weight_table(%arg2 : !WeightsTableStub_CMX)
                         activation_window(%arg3 : !ActWindStub_CMX)
                         parent_input(%arg0 : !InputStub_CMX)
                         parent_output(%arg4 : !OutputStub_CMX)
                         outputs(%arg4 : !OutputStub_CMX)
                             -> !OutputStub_CMX variants :  {
                            DPUTask {
                                start = [0, 0, 0], end = [31, 31, 15],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                start = [0, 0, 16], end = [31, 31, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 1 : i64
                            }
                         } PPE :  {
                         }
        }
    }

    return %parent_out_cmx: !ParentOutputDistributed

    // CHECK:       [[WEIGHTS_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[WEIGHTS_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = SEGMENTED, num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[ACT_WIN_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <[[ACT_WIN_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer<16x1x1x16xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = SEGMENTED, num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHT_TABLE_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" <33280> -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = SEGMENTED, num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[CONST:%.*]] = const.Declare memref<32x1x1x4xsi32> = #const.Content<dense<1> : tensor<32x1x1x4xsi32>, [#const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, [[ACT_WIN_ADDR]] : i64, [0, 16]>]>

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[CONST]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF]]
    // CHECK-NEXT:          VPUIP.NNDMA
}

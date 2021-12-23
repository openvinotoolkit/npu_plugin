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
    // CHECK:       [[CONST:%.*]] = const.Declare memref<1008x1x1x4xsi32> = #const.Content<dense<1> : tensor<1008x1x1x4xsi32>, [#const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, [[ACT_WIN_ADDR]] : i64>]>
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
    // CHECK:       [[CONST:%.*]] = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>, [#const.RelocateWeightsTable<0 : i64, [[ACT_WIN_ADDR]] : i64>]>
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
    // CHECK:       [[CONST:%.*]] = const.Declare memref<256x1x1x1xsi32> = #const.Content<dense<1> : tensor<256x1x1x1xsi32>, [#const.RelocateWeightsTable<[[WEIGHTS_ADDR]] : i64, 16777215 : i64>]>
    // CHECK:       [[NDMA_OP:.*]] = VPUIP.NNDMA {port = 0 : i64} inputs([[CONST]] : memref<256x1x1x1xsi32>) outputs([[WEIGHT_TABLE_BUF]] : memref<256x1x1x1xsi32, @CMX_NN>) -> memref<256x1x1x1xsi32, @CMX_NN>
    // CHECK:       [[NCE_CLUST_TASK_OP:.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  weights([[WEIGHTS_BUF]] : memref<64x16x3x3xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE_BUF]] : memref<256x1x1x1xsi32, @CMX_NN>)
}


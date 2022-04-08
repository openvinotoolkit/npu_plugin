// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --swizzle-constant %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @SwizzleConstants() {
    %cst = const.Declare memref<16x16x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>]
    
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<16x16x1x1xf16, #NHWC>) outputs(%1 : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    return

    // CHECK: [[WT_CONST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC> = dense<1.000000e+00> : 
    // CHECK-SAME: tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>, #const.SwizzleConstant<5 : i64, 3 : i64>]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwizzleConstantsSparseWeights
func @SwizzleConstantsSparseWeights() -> memref<256x1x1x1xsi32, [@CMX_NN, 0]> {
    %weight_table_const = const.Declare memref<256x1x1x1xsi32> = dense<1> : tensor<256x1x1x1xsi32>
    %weights_const = const.Declare memref<64x16x3x3xf16, #NHWC> = dense<1.0> : tensor<64x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm_const = const.Declare memref<64x1x1x256xi1> = dense<1.0> : tensor<64x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    %weight_table = VPURT.DeclareBuffer "CMX_NN" [0] <1024> {swizzlingKey = 5 : i64} -> memref<256x1x1x1xsi32, [@CMX_NN, 0]>
    %weights = VPURT.DeclareBuffer "CMX_NN" [0] <4096> {swizzlingKey = 5 : i64} -> memref<64x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %weights_sm = VPURT.DeclareBuffer "CMX_NN" [0] <22528> {swizzlingKey = 5 : i64} -> memref<64x1x1x256xi1, [@CMX_NN, 0]>

    %in = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x59x227xf16, #NHWC, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer "CMX_NN" [0] <428608> -> memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>

    %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%weights_const : memref<64x16x3x3xf16, #NHWC>) outputs(%weights : memref<64x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%weights_sm_const : memref<64x1x1x256xi1>) outputs(%weights_sm : memref<64x1x1x256xi1, [@CMX_NN, 0]>) -> memref<64x1x1x256xi1, [@CMX_NN, 0]>
    %5 = VPUIP.NNDMA {port = 0 : i64} inputs(%weight_table_const : memref<256x1x1x1xsi32>) outputs(%weight_table : memref<256x1x1x1xsi32, [@CMX_NN, 0]>) -> memref<256x1x1x1xsi32, [@CMX_NN, 0]>

    %6 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = "CONV"}
        input(%in : memref<1x16x59x227xf16, #NHWC, [@CMX_NN, 0]>)
        weights(%weights : memref<64x16x3x3xf16, #NHWC, [@CMX_NN, 0]>)
        weights_sparsity_map(%weights_sm : memref<64x1x1x256xi1, [@CMX_NN, 0]>)
        weight_table(%weight_table : memref<256x1x1x1xsi32, [@CMX_NN, 0]>)
        parent_input(%in : memref<1x16x59x227xf16, #NHWC, [@CMX_NN, 0]>)
        parent_output(%out : memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%out : memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>
      variants : {
        DPUTask {end = [112, 4, 63], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    } PPE :  {
    }

    return %weight_table : memref<256x1x1x1xsi32, [@CMX_NN, 0]>

    // CHECK: [[WT_CONST:%.*]] = const.Declare memref<256x1x1x1xsi32> = dense<1> : tensor<256x1x1x1xsi32>, [#const.SwizzleConstant<5 : i64, 3 : i64>]
    // CHECK: [[W_CONST:%.*]] = const.Declare memref<64x16x3x3xf16, #NHWC> = dense<1.000000e+00> : tensor<64x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>, #const.SwizzleConstant<5 : i64, 3 : i64>]
    // CHECK: [[W_SM_CONST:%.*]] = const.Declare memref<64x1x1x256xi1> = dense<1.000000e+00> : tensor<64x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap, #const.SwizzleConstant<5 : i64, 3 : i64>]

    // CHECK: [[WEIGHT_TABLE_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <1024> {swizzlingKey = 5 : i64} -> memref<256x1x1x1xsi32, [@CMX_NN, 0]>
    // CHECK: [[WEIGHTS_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> {swizzlingKey = 5 : i64} -> memref<64x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[WEIGHTS_SM_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <22528> {swizzlingKey = 5 : i64} -> memref<64x1x1x256xi1, [@CMX_NN, 0]>

    // CHECK: [[NDMA_OP_W:.*]] = VPUIP.NNDMA {port = 0 : i64} inputs([[W_CONST]] : memref<64x16x3x3xf16, #NHWC>) outputs([[WEIGHTS_BUF]] : memref<64x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[NDMA_OP_W_SM:.*]] = VPUIP.NNDMA {port = 0 : i64} inputs([[W_SM_CONST]] : memref<64x1x1x256xi1>) outputs([[WEIGHTS_SM_BUF]] : memref<64x1x1x256xi1, [@CMX_NN, 0]>) -> memref<64x1x1x256xi1, [@CMX_NN, 0]>
    // CHECK: [[NDMA_OP_WT:.*]] = VPUIP.NNDMA {port = 0 : i64} inputs([[WT_CONST]] : memref<256x1x1x1xsi32>) outputs([[WEIGHT_TABLE_BUF]] : memref<256x1x1x1xsi32, [@CMX_NN, 0]>) -> memref<256x1x1x1xsi32, [@CMX_NN, 0]>

    // CHECK: [[NCE_CLUST_TASK_OP:.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME: weights([[WEIGHTS_BUF]] : memref<64x16x3x3xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME: weights_sparsity_map([[WEIGHTS_SM_BUF]] : memref<64x1x1x256xi1, [@CMX_NN, 0]>)
    // CHECK-SAME: weight_table([[WEIGHT_TABLE_BUF]] : memref<256x1x1x1xsi32, [@CMX_NN, 0]>)
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SwizzleFusedConstants
func @SwizzleFusedConstants() -> memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]> {
    %fused_const = const.Declare memref<1x1x1x19456xui8> = dense<1> : tensor<1x1x1x19456xui8>

    %fused_buffer = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> memref<1x1x1x19456xui8, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>
    %weight_table = VPURT.DeclareBuffer "CMX_NN" [0] <1024> {swizzlingKey = 5 : i64} -> memref<256x1x1x1xsi32, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>
    %weights = VPURT.DeclareBuffer "CMX_NN" [0] <4096> {swizzlingKey = 5 : i64} -> memref<64x16x3x3xf16, {order = #NHWC, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>

    %in = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x59x227xf16, #NHWC, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>

    %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%fused_const : memref<1x1x1x19456xui8, #NCHW>) outputs(%fused_buffer : memref<1x1x1x19456xui8, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>) -> memref<1x1x1x19456xui8, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>

    %6 = VPUIP.NCEClusterTask {kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = "CONV"}
        input(%in : memref<1x16x59x227xf16, #NHWC, [@CMX_NN, 0]>)
        weights(%weights : memref<64x16x3x3xf16, {order = #NHWC, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>)
        weight_table(%weight_table : memref<256x1x1x1xsi32, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>)
        parent_input(%in : memref<1x16x59x227xf16, #NHWC, [@CMX_NN, 0]>)
        parent_output(%out : memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%out : memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>
      variants : {
        DPUTask {end = [112, 4, 63], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
    } PPE :  {
    }

    return %6: memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK: [[FUSED_CONST:%.*]] = const.Declare memref<1x1x1x19456xui8> = dense<1> : tensor<1x1x1x19456xui8>, [#const.SwizzleConstant<5 : i64, 3 : i64>]

    // CHECK: [[FUSED_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> memref<1x1x1x19456xui8, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>
    // CHECK: [[WEIGHT_TABLE_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <1024> {swizzlingKey = 5 : i64} -> memref<256x1x1x1xsi32, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>
    // CHECK: [[WEIGHTS_BUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <4096> {swizzlingKey = 5 : i64} -> memref<64x16x3x3xf16, {order = #NHWC, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>

    // CHECK: VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x59x227xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: VPURT.DeclareBuffer "CMX_NN" [0] <32768> -> memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK: VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<1x1x1x19456xui8>) outputs(%0 : memref<1x1x1x19456xui8, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>) -> memref<1x1x1x19456xui8, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>

    // CHECK: [[NCE_CLUST_TASK_OP:.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME: weights([[WEIGHTS_BUF]] : memref<64x16x3x3xf16, {order = #NHWC, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>)
    // CHECK-SAME: weight_table([[WEIGHT_TABLE_BUF]] : memref<256x1x1x1xsi32, {order = #NCHW, swizzlingKey = 5 : i64}, [@CMX_NN, 0]>)
}

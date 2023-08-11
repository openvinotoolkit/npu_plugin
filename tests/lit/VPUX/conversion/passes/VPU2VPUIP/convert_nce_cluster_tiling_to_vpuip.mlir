//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-nce-cluster-tiling-to-vpuip --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!type_DDR_tensor = tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!type_DDR_memref = memref<1x32x16x16xf16, #NHWC, @DDR>
!type_CMX_tensor = tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!type_CMX_memref = memref<1x32x16x16xf16, #NHWC, @CMX_NN>

// NCEClusterTiling operation with memref output
// Original operation before IE2IERT lowering:
// func.func @NCEClusterTilingCopyOpTensorResult(%input0: !type_DDR_tensor) -> !type_CMX_tensor{
//     %tensor_cmx = VPU.NCE.ClusterTiling(%input0 as %arg0: !type_DDR_tensor) -> !type_CMX_tensor {
//         %0 = IE.Copy(%arg0) { out_mem_space = @CMX_NN } : !type_DDR_tensor -> !type_CMX_tensor
//         VPU.Yield %0
//     }
//     return %tensor_cmx : !type_CMX_tensor
// }

// CHECK-LABEL: @NCEClusterTilingCopyOpTensorResult
func.func @NCEClusterTilingCopyOpTensorResult(%input0: !type_DDR_memref) -> !type_CMX_memref{

    %input_DDR_tensor = builtin.unrealized_conversion_cast %input0 : !type_DDR_memref
        to !type_DDR_tensor

    %tensor_cmx = VPU.NCE.ClusterTiling(%input_DDR_tensor as %arg0: !type_DDR_tensor) -> !type_CMX_tensor {
        %input_DDR_memref = builtin.unrealized_conversion_cast %arg0: !type_DDR_tensor
            to !type_DDR_memref
        %0 = memref.alloc() : !type_CMX_memref
        %1 = VPUIP.Copy inputs(%input_DDR_memref : !type_DDR_memref) outputs(%0 : !type_CMX_memref) -> !type_CMX_memref
        %2 = builtin.unrealized_conversion_cast %1 : !type_CMX_memref to !type_CMX_tensor
        VPU.Yield %2
    }

    %memref_cmx = builtin.unrealized_conversion_cast %tensor_cmx : !type_CMX_tensor
        to !type_CMX_memref

    return %memref_cmx : !type_CMX_memref
}

// CHECK:       [[ALLOC0:%.+]] = memref.alloc() : memref<1x32x16x16xf16, #NHWC, @CMX_NN>
// CHECK:       [[NCE_CT_RES:%.+]] = VPUIP.NCEClusterTiling
// CHECK-SAME:       inputs(%arg0 as [[ARG1:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
// CHECK-SAME:       outputs([[ALLOC0]] as [[ARG2:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:       -> memref<1x32x16x16xf16, #NHWC, @CMX_NN>
// CHECK:       [[COPY_RES:%.+]] = VPUIP.Copy
// CHECK-SAME:       inputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, @DDR>)
// CHECK-SAME:      outputs([[ARG2]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!typeCmxDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = {bottom = 1, left = 1, right = 1, top = 1},
    strides = [1, 1],
    num_clusters = 4
}>

!type_DDR_tensor = tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!type_DDR_memref = memref<1x32x16x16xf16, #NHWC, @DDR>
!type_CMX_tensor = tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!type_CMX_memref = memref<1x32x16x16xf16, #NHWC, @CMX_NN>

// NCEClusterTiling operation with distributed type of output
// Original operation before IE2IERT lowering
// func.func @NCEClusterTilingCopyOpDistributedResult(%input0: !type_DDR_tensor) -> !type_CMX_tensor{
//     %tensor_distributed_cmx = VPU.NCE.ClusterTiling(%input0 as %arg0: !type_DDR_tensor) -> !typeCmxDistributed {
//         %0 = IE.Copy(%arg0) { out_mem_space = @CMX_NN } : !type_DDR_tensor -> !type_CMX_tensor
//         VPU.Yield %0
//     }
//    %tensor_cmx = builtin.unrealized_conversion_cast %tensor_distributed_cmx : !typeCmxDistributed
//         to !type_CMX_tensor
//     return %tensor_cmx : !type_CMX_tensor
// }
// CHECK-LABEL: @NCEClusterTilingCopyOpDistributedResult
func.func @NCEClusterTilingCopyOpDistributedResult(%input0: !type_DDR_memref) -> !type_CMX_memref{

    %input_DDR_tensor = builtin.unrealized_conversion_cast %input0 : !type_DDR_memref
        to !type_DDR_tensor

    %tensor_distributed_cmx = VPU.NCE.ClusterTiling(%input_DDR_tensor as %arg0: !type_DDR_tensor) -> !typeCmxDistributed {
        %input_DDR_memref = builtin.unrealized_conversion_cast %arg0: !type_DDR_tensor
            to !type_DDR_memref
        %0 = memref.alloc() : !type_CMX_memref
        %1 = VPUIP.Copy inputs(%input_DDR_memref : !type_DDR_memref) outputs(%0 : !type_CMX_memref) -> !type_CMX_memref
        %2 = builtin.unrealized_conversion_cast %1 : !type_CMX_memref to !type_CMX_tensor
        VPU.Yield %2
    }

    %memref_cmx = builtin.unrealized_conversion_cast %tensor_distributed_cmx : !typeCmxDistributed
        to !type_CMX_memref

    return %memref_cmx : !type_CMX_memref
}

// CHECK:       [[ALLOC0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
// CHECK:       [[NCE_CT_RES:%.+]] = VPUIP.NCEClusterTiling
// CHECK-SAME:       inputs(%arg0 as [[ARG1:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
// CHECK-SAME:       outputs([[ALLOC0]] as [[ARG2:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:       -> !VPUIP.DistributedBuffer
// CHECK:       [[COPY_RES:%.+]] = VPUIP.Copy
// CHECK-SAME:       inputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, @DDR>)
// CHECK-SAME:      outputs([[ARG2]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!typeCmxDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = {bottom = 1, left = 1, right = 1, top = 1},
    strides = [1, 1],
    num_clusters = 4
}>

!type_DDR_tensor = tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!type_DDR_memref = memref<1x32x16x16xf16, #NHWC, @DDR>
!type_CMX_tensor = tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!type_CMX_memref = memref<1x32x16x16xf16, #NHWC, @CMX_NN>

// 2 NCEClusterTiling operations with distributed type passed in between
// Original operation before IE2IERT lowering
// func.func @NCEClusterTilingDistributedCopy2CopyOp(%input0: !type_DDR_tensor) -> !type_DDR_tensor {
//     %tensor_distributed_cmx = VPU.NCE.ClusterTiling(%input0 as %arg0: !type_DDR_tensor) -> !typeCmxDistributed {
//         %0 = IE.Copy(%arg0) { out_mem_space = @CMX_NN } : !type_DDR_tensor -> !type_CMX_tensor
//         VPU.Yield %0
//     }
//     %tensor_ddr = VPU.NCE.ClusterTiling(%tensor_distributed_cmx as %arg0: !type_CMX_tensor) -> !type_DDR_tensor {
//         %0 = IE.Copy(%arg0) { out_mem_space = @DDR } : !type_CMX_tensor -> !type_DDR_tensor
//         VPU.Yield %0
//     }
//     return %tensor_ddr : !type_DDR_tensor
// }

// CHECK-LABEL: @NCEClusterTilingDistributedCopy2CopyOp
func.func @NCEClusterTilingDistributedCopy2CopyOp(%arg0: !type_DDR_memref, %arg1: !type_DDR_memref) -> !type_DDR_memref {
    %0 = builtin.unrealized_conversion_cast %arg0 : !type_DDR_memref to !type_DDR_tensor
    %1 = VPU.NCE.ClusterTiling (%0 as %arg2: !type_DDR_tensor) -> !typeCmxDistributed {
        %5 = builtin.unrealized_conversion_cast %arg2 : !type_DDR_tensor to !type_DDR_memref
        %6 = memref.alloc() : !type_CMX_memref
        %7 = VPUIP.Copy inputs(%5 : !type_DDR_memref) outputs(%6 : !type_CMX_memref) -> !type_CMX_memref
        %8 = builtin.unrealized_conversion_cast %7 : !type_CMX_memref to !type_CMX_tensor
        VPU.Yield %8
    }
    %2 = VPU.NCE.ClusterTiling (%1 as %arg2: !type_CMX_tensor) -> !type_DDR_tensor {
        %5 = builtin.unrealized_conversion_cast %arg2 : !type_CMX_tensor to !type_CMX_memref
        %6 = memref.alloc() : !type_DDR_memref
        %7 = VPUIP.Copy inputs(%5 : !type_CMX_memref) outputs(%6 : !type_DDR_memref) -> !type_DDR_memref
        %8 = builtin.unrealized_conversion_cast %7 : !type_DDR_memref to !type_DDR_tensor
        VPU.Yield %8
    }
    %3 = builtin.unrealized_conversion_cast %2 : !type_DDR_tensor to !type_DDR_memref
    %4 = VPUIP.Copy inputs(%3 : !type_DDR_memref) outputs(%arg1 : !type_DDR_memref) -> !type_DDR_memref
    return %4 : !type_DDR_memref
}

// CHECK:       [[ALLOC0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
// CHECK:       [[NCE_CT_RES0:%.+]] = VPUIP.NCEClusterTiling
// CHECK-SAME:       inputs(%arg0 as [[ARG1:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
// CHECK-SAME:       outputs([[ALLOC0]] as [[ARG2:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:       -> !VPUIP.DistributedBuffer
// CHECK:       [[COPY_RES0:%.+]] = VPUIP.Copy
// CHECK-SAME:       inputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, @DDR>)
// CHECK-SAME:      outputs([[ARG2]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
// CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x32x16x16xf16, #NHWC, @DDR>
// CHECK:       [[NCE_CT_RES1:%.+]] = VPUIP.NCEClusterTiling
// CHECK-SAME:       inputs([[NCE_CT_RES0]] as [[ARG3:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:       outputs([[ALLOC1]] as [[ARG4:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
// CHECK-SAME:       -> memref<1x32x16x16xf16, #NHWC, @DDR>
// CHECK:       [[COPY_RES1:%.+]] = VPUIP.Copy
// CHECK-SAME:       inputs([[ARG3]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      outputs([[ARG4]] : memref<1x32x16x16xf16, #NHWC, @DDR>)

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!TensorDistributed = !VPU.DistributedTensor<
    32x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!SMTensorDistributed = !VPU.DistributedTensor<
    32x1x1x256xi1, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!BufferDistributed = !VPUIP.DistributedBuffer<
    32x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!SMBufferDistributed = !VPUIP.DistributedBuffer<
    32x1x1x256xi1, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!Tensor_DDR = tensor<32x16x3x3xf16, {mem_space = @DDR, order = #NHWC}>
!SMTensor_DDR = tensor<32x1x1x256xi1, {mem_space = @DDR}>
!Tensor_CMX = tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!SMTensor_CMX = tensor<32x1x1x256xi1, {mem_space = @CMX_NN}>

!Buffer_DDR = memref<32x16x3x3xf16, #NHWC, @DDR>
!SMBuffer_DDR = memref<32x1x1x256xi1, @DDR>
!Buffer_CMX = memref<32x16x3x3xf16, #NHWC, @CMX_NN>
!SMBuffer_CMX = memref<32x1x1x256xi1, @CMX_NN>

// CHECK:  func.func @SparseDistributedCopyCMXToDDR([[ARG0:%.+]]: memref<32x16x3x3xf16, #NHWC, @DDR>
func.func @SparseDistributedCopyCMXToDDR(%arg0: !Buffer_DDR, %arg1: !VPUIP.SparseBuffer<data=!BufferDistributed, sparsity_map=!SMBufferDistributed, is_weights>)
        -> !VPUIP.SparseBuffer<data=!BufferDistributed, sparsity_map=!SMBufferDistributed, is_weights> {

    %cst_sm = const.Declare !SMBuffer_DDR = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %0 = VPUIP.GroupSparseBuffer(%arg0, %cst_sm) {is_weights} -> !VPUIP.SparseBuffer<data=!Buffer_DDR, sparsity_map=!SMBuffer_DDR, is_weights>

    %1 = builtin.unrealized_conversion_cast %0 : !VPUIP.SparseBuffer<data=!Buffer_DDR, sparsity_map=!SMBuffer_DDR, is_weights>
            to !VPU.SparseTensor<data=!Tensor_DDR, sparsity_map=!SMTensor_DDR, is_weights>

    %2 = VPU.NCE.ClusterTiling (%1 as %arg2: !VPU.SparseTensor<data=!Tensor_DDR, sparsity_map=!SMTensor_DDR, is_weights>)
            -> !VPU.SparseTensor<data=!TensorDistributed, sparsity_map=!SMTensorDistributed, is_weights> {
        %4 = builtin.unrealized_conversion_cast %arg2 : !VPU.SparseTensor<data=!Tensor_DDR, sparsity_map=!SMTensor_DDR, is_weights>
                to !VPUIP.SparseBuffer<data=!Buffer_DDR, sparsity_map=!SMBuffer_DDR, is_weights>
        %5 = memref.alloc() : !Buffer_CMX
        %6 = memref.alloc() : !SMBuffer_CMX
        %7 = VPUIP.GroupSparseBuffer(%5, %6) {is_weights} -> !VPUIP.SparseBuffer<data=!Buffer_CMX, sparsity_map=!SMBuffer_CMX, is_weights>
        %8 = VPUIP.Copy inputs(%4 : !VPUIP.SparseBuffer<data=!Buffer_DDR, sparsity_map=!SMBuffer_DDR, is_weights>)
                        outputs(%7 : !VPUIP.SparseBuffer<data=!Buffer_CMX, sparsity_map=!SMBuffer_CMX, is_weights>)
                -> !VPUIP.SparseBuffer<data=!Buffer_CMX, sparsity_map=!SMBuffer_CMX, is_weights>
        %9 = builtin.unrealized_conversion_cast %8 : !VPUIP.SparseBuffer<data=!Buffer_CMX, sparsity_map=!SMBuffer_CMX, is_weights>
                to !VPU.SparseTensor<data=!Tensor_CMX, sparsity_map=!SMTensor_CMX, is_weights>
        VPU.Yield %9
    }

    %3 = builtin.unrealized_conversion_cast %2 : !VPU.SparseTensor<data=!TensorDistributed, sparsity_map=!SMTensorDistributed, is_weights>
            to !VPUIP.SparseBuffer<data=!BufferDistributed, sparsity_map=!SMBufferDistributed, is_weights>

    return %3 : !VPUIP.SparseBuffer<data=!BufferDistributed, sparsity_map=!SMBufferDistributed, is_weights>

    // CHECK-DAG:       [[CST_SM:%.+]] = const.Declare memref<32x1x1x256xi1, @DDR> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:       [[INPUT_SPARSE_DDR:%.+]] = VPUIP.GroupSparseBuffer(%arg0, [[CST_SM]]) {is_weights}
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, #NHWC, @DDR>, sparsity_map=memref<32x1x1x256xi1, @DDR>, is_weights>
    // CHECK:       [[DATA_DIST_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[SM_DIST_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x1x1x256xi1, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUTPUT_SPARSE_CMX:%.+]] = VPUIP.GroupSparseBuffer([[DATA_DIST_CMX]], [[SM_DIST_CMX]]) {is_weights}
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                 sparsity_map=!VPUIP.DistributedBuffer<32x1x1x256xi1, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                 is_weights>
    // CHECK:       [[OUTPUT_SPARSE_COPY_CMX:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[INPUT_SPARSE_DDR]] as [[ARG2:%.+]]: !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, #NHWC, @DDR>, sparsity_map=memref<32x1x1x256xi1, @DDR>, is_weights>)
    // CHECK-SAME:      outputs([[OUTPUT_SPARSE_CMX]] as [[ARG3:%.+]]: !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, #NHWC, @CMX_NN>, sparsity_map=memref<32x1x1x256xi1, @CMX_NN>, is_weights>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                 sparsity_map=!VPUIP.DistributedBuffer<32x1x1x256xi1, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                 is_weights> {
    // CHECK:           [[COPY_OUT:%.+]] = VPUIP.Copy inputs([[ARG2]]
    // CHECK-SAME:                                    outputs([[ARG3]]
    // CHECK:       }
    // CHECK:       return [[OUTPUT_SPARSE_COPY_CMX]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IOTensorDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!IOSMTensorDistributed = !VPU.DistributedTensor<
    1x32x16x16xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!IOBufferDistributed = !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!IOSMBufferDistributed = !VPUIP.DistributedBuffer<
    1x32x16x16xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!IOTensor = tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!IOSMTensor = tensor<1x32x16x16xi1, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTensor = tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsSMTensor = tensor<64x1x1x384xi1, {mem_space = @CMX_NN}>
!WeightsTableTensor = tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

!IOBuffer = memref<1x32x16x16xf16, #NHWC, @CMX_NN>
!IOSMBuffer = memref<1x32x16x16xi1, #NHWC, @CMX_NN>
!WeightsBuffer = memref<64x32x3x3xf16, #NHWC, @CMX_NN>
!WeightsSMBuffer = memref<64x1x1x384xi1, @CMX_NN>
!WeightsTableBuffer = memref<64x1x1x4xsi32, @CMX_NN>

// CHECK:  func.func @SparseDistributedNCEConv([[ARG0:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>, [[ARG1:%.+]]: memref<1x32x16x16xi1, #NHWC, @CMX_NN>
func.func @SparseDistributedNCEConv(%arg0: !IOBuffer, %arg1: !IOSMBuffer, %arg2: !VPUIP.SparseBuffer<data=!IOBufferDistributed, sparsity_map=!IOSMBufferDistributed>)
        -> !VPUIP.SparseBuffer<data=!IOBufferDistributed, sparsity_map=!IOSMBufferDistributed> {

    %input_sparse = VPUIP.GroupSparseBuffer(%arg0, %arg1) -> !VPUIP.SparseBuffer<data=!IOBuffer, sparsity_map=!IOSMBuffer>
    %input_sparse_tensor = builtin.unrealized_conversion_cast %input_sparse : !VPUIP.SparseBuffer<data=!IOBuffer, sparsity_map=!IOSMBuffer>
            to !VPU.SparseTensor<data=!IOTensor, sparsity_map=!IOSMTensor>

    %cst_weights = const.Declare !WeightsBuffer = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare !WeightsSMBuffer = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPUIP.GroupSparseBuffer(%cst_weights, %cst_weights_sm) {is_weights} -> !VPUIP.SparseBuffer<data=!WeightsBuffer, sparsity_map=!WeightsSMBuffer, is_weights>
    %weights_sparse_tensor = builtin.unrealized_conversion_cast %weights_sparse : !VPUIP.SparseBuffer<data=!WeightsBuffer, sparsity_map=!WeightsSMBuffer, is_weights>
            to !VPU.SparseTensor<data=!WeightsTensor, sparsity_map=!WeightsSMTensor, is_weights>

    %cst_weights_table = const.Declare !WeightsTableBuffer = dense<1> : tensor<64x1x1x4xsi32>
    %weights_table_tensor = builtin.unrealized_conversion_cast %cst_weights_table : !WeightsTableBuffer to !WeightsTableTensor

    %output_sparse_tensor = VPU.NCE.ClusterTiling (
                %input_sparse_tensor as %arg3: !VPU.SparseTensor<data=!IOTensor, sparsity_map=!IOSMTensor>,
                %weights_sparse_tensor as %arg4: !VPU.SparseTensor<data=!WeightsTensor, sparsity_map=!WeightsSMTensor, is_weights>,
                %weights_table_tensor as %arg5: !WeightsTableTensor)
            -> !VPU.SparseTensor<data=!IOTensorDistributed, sparsity_map=!IOSMTensorDistributed> {

        %in_sparse_buffer = builtin.unrealized_conversion_cast %arg3 : !VPU.SparseTensor<data=!IOTensor, sparsity_map=!IOSMTensor>
                to !VPUIP.SparseBuffer<data=!IOBuffer, sparsity_map=!IOSMBuffer>
        %w_sparse_buffer = builtin.unrealized_conversion_cast %arg4 : !VPU.SparseTensor<data=!WeightsTensor, sparsity_map=!WeightsSMTensor, is_weights>
                to !VPUIP.SparseBuffer<data=!WeightsBuffer, sparsity_map=!WeightsSMBuffer, is_weights>
        %wt_buffer = builtin.unrealized_conversion_cast %arg5 : !WeightsTableTensor to !WeightsTableBuffer

        %in_data, %in_sm = VPUIP.UngroupSparseBuffer(%in_sparse_buffer) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> !IOBuffer, !IOSMBuffer
        %w_data, %w_sm = VPUIP.UngroupSparseBuffer(%w_sparse_buffer) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} -> !WeightsBuffer, !WeightsSMBuffer
        %out_data = memref.alloc() : !IOBuffer
        %out_sm = memref.alloc() : !IOSMBuffer

        %out:2 = VPUIP.NCEClusterTask {
                    kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
                    kernel_size = [3, 3],
                    kernel_strides = [1, 1],
                    task_type = "CONV"
                }
                    input(%in_data : !IOBuffer)
                    input_sparsity_map(%in_sm : !IOSMBuffer)
                    weights(%w_data : !WeightsBuffer)
                    weights_sparsity_map(%w_sm : !WeightsSMBuffer)
                    weight_table(%wt_buffer : !WeightsTableBuffer)
                    parent_input(%in_data : !IOBuffer)
                    parent_input_sparsity_map(%in_sm : !IOSMBuffer)
                    parent_output(%out_data : !IOBuffer)
                    parent_output_sparsity_map(%out_sm : !IOSMBuffer)
                    outputs(%out_data : !IOBuffer)
                    output_sparsity_map(%out_sm : !IOSMBuffer)
                -> !IOBuffer, !IOSMBuffer
                variants : {
                    DPUTask {outEnd = [15, 15, 31], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
                } PPE : {
                }
        %out_sparse_buffer = VPUIP.GroupSparseBuffer(%out#0, %out#1) -> !VPUIP.SparseBuffer<data=!IOBuffer, sparsity_map=!IOSMBuffer>
        %out_sparse_tensor = builtin.unrealized_conversion_cast %out_sparse_buffer : !VPUIP.SparseBuffer<data=!IOBuffer, sparsity_map=!IOSMBuffer>
                to !VPU.SparseTensor<data=!IOTensor, sparsity_map=!IOSMTensor>
        VPU.Yield %out_sparse_tensor
    }

    %output_sparse_buffer = builtin.unrealized_conversion_cast %output_sparse_tensor : !VPU.SparseTensor<data=!IOTensorDistributed, sparsity_map=!IOSMTensorDistributed>
            to !VPUIP.SparseBuffer<data=!IOBufferDistributed, sparsity_map=!IOSMBufferDistributed>

    return %output_sparse_buffer : !VPUIP.SparseBuffer<data=!IOBufferDistributed, sparsity_map=!IOSMBufferDistributed>

    // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare memref<64x32x3x3xf16, #NHWC, @CMX_NN> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:   [[CST_WEIGHTS_SM:%.+]] = const.Declare memref<64x1x1x384xi1, @CMX_NN> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK-DAG:   [[CST_WEIGHTS_TABLE:%.+]] = const.Declare memref<64x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<64x1x1x4xsi32>

    // CHECK:       [[OUTPUT_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUTPUT_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CONV_OUTPUT:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[ARG0]] as [[ARG3:%[^:]+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:             [[ARG1]] as [[ARG4:%[^:]+]]: memref<1x32x16x16xi1, #NHWC, @CMX_NN>,
    // CHECK-SAME:             [[CST_WEIGHTS]] as [[ARG5:%[^:]+]]: memref<64x32x3x3xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:             [[CST_WEIGHTS_SM]] as [[ARG6:%[^:]+]]: memref<64x1x1x384xi1, @CMX_NN>,
    // CHECK-SAME:             [[CST_WEIGHTS_TABLE]] as [[ARG7:%[^:]+]]: memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUTPUT_DATA]] as [[ARG8:%[^:]+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:              [[OUTPUT_SM]] as [[ARG9:%[^:]+]]: memref<1x32x16x16xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:      -> (!VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:           [[CONV_OUT:%.+]]:2 = VPUIP.NCEClusterTask {
    // CHECK-SAME:          kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          kernel_size = [3, 3],
    // CHECK-SAME:          kernel_strides = [1, 1],
    // CHECK-SAME:          task_type = "CONV"
    // CHECK-SAME:      }
    // CHECK-SAME:          input([[ARG3]]
    // CHECK-SAME:          input_sparsity_map([[ARG4]]
    // CHECK-SAME:          weights([[ARG5]]
    // CHECK-SAME:          weights_sparsity_map([[ARG6]]
    // CHECK-SAME:          weight_table([[ARG7]]
    // CHECK-SAME:          parent_input([[ARG3]]
    // CHECK-SAME:          parent_input_sparsity_map([[ARG4]]
    // CHECK-SAME:          parent_output([[ARG8]]
    // CHECK-SAME:          parent_output_sparsity_map([[ARG9]]
    // CHECK-SAME:          outputs([[ARG8]]
    // CHECK-SAME:          output_sparsity_map([[ARG9]]
    // CHECK-SAME:      -> memref<1x32x16x16xf16, #NHWC, @CMX_NN>, memref<1x32x16x16xi1, #NHWC, @CMX_NN>
    // CHECK-SAME:      variants : {
    // CHECK:               DPUTask {mpe_mode = "VECTOR_FP16", outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    // CHECK:           } PPE : {
    // CHECK:           }
    // CHECK:       }

    // CHECK:       [[OUTPUT_SPARSE:%.+]] = VPUIP.GroupSparseBuffer([[CONV_OUTPUT]]#0, [[CONV_OUTPUT]]#1) -> !VPUIP.SparseBuffer<
    // CHECK-SAME:      data=!VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:      sparsity_map=!VPUIP.DistributedBuffer<1x32x16x16xi1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>>

    // CHECK:       return [[OUTPUT_SPARSE]]
}


// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

!type_CMX_tensor = tensor<1x4x512x1xf16, {mem_space = @CMX_NN, order = #NWHC}>
!type_CMX_memref = memref<1x4x512x1xf16, #NWHC, @CMX_NN>


!typeCmxDistributed = !VPU.DistributedTensor<
    1x4x512x1xf16, #NWHC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

module @VPU.SW {
func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}


func.func @NCEClusterTilingWithSWOp(%arg0: !typeCmxDistributed) -> !typeCmxDistributed {

 %184 = VPU.NCE.ClusterTiling (%arg0 as %arg2: !type_CMX_tensor) -> !typeCmxDistributed {
      %205 = builtin.unrealized_conversion_cast %arg2 : !type_CMX_tensor to !type_CMX_memref
      %206 = memref.alloc() : !type_CMX_memref
      %results_33 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%205 as %arg3: !type_CMX_memref) outputs(%206 as %arg4: !type_CMX_memref) on tile 0 -> !type_CMX_memref {
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : !type_CMX_memref, !type_CMX_memref
      }
      %207 = builtin.unrealized_conversion_cast %results_33 : !type_CMX_memref to !type_CMX_tensor
      VPU.Yield %207
    }
    return %184 : !typeCmxDistributed
}

// CHECK:  [[ARG0:%.+]] = builtin.unrealized_conversion_cast %arg0 : !VPU.DistributedTensor<1x4x512x1xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x4x512x1xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:  [[ARG1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x512x1xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:  [[ARG2:%.+]] = VPUIP.NCEClusterTiling inputs([[ARG0]] as [[inner_arg1:%.+]]: memref<1x4x512x1xf16, #NWHC, @CMX_NN>) outputs([[ARG1]] as [[inner_arg2:%.+]]: memref<1x4x512x1xf16, #NWHC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x4x512x1xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}
// CHECK:      [[results:%.+]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs([[inner_arg1]] as [[inner_arg3:%.+]]: memref<1x4x512x1xf16, #NWHC, @CMX_NN>) outputs([[inner_arg2]] as [[inner_arg4:%.+]]: memref<1x4x512x1xf16, #NWHC, @CMX_NN>) on tile 0 -> memref<1x4x512x1xf16, #NWHC, @CMX_NN>{
// CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}([[inner_arg3]], [[inner_arg4]]) : memref<1x4x512x1xf16, #NWHC, @CMX_NN>, memref<1x4x512x1xf16, #NWHC, @CMX_NN>
// CHECK:      }
// CHECK:    }
// CHECK:  [[ARG3:%.+]] = builtin.unrealized_conversion_cast [[ARG2]] : !VPUIP.DistributedBuffer<1x4x512x1xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPU.DistributedTensor<1x4x512x1xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:  return [[ARG3]] : !VPU.DistributedTensor<1x4x512x1xf16, #NWHC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OTensorDistributed = !VPU.DistributedTensor<
    1x64x112x112xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 :i64
}>

!in_CMX_tensor = tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
!weight_CMX_tensor = tensor<64x1x1x160xf16, {mem_space = @CMX_NN, order = #NHWC}>
!weight_table_CMX_tensor = tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

// CHECK-LABEL: @NCEClusterTilingCompressConv
func.func @NCEClusterTilingCompressConv(%input: !in_CMX_tensor, %in_weights: !weight_CMX_tensor, %in_weights_table: !weight_table_CMX_tensor) -> !OTensorDistributed{
    %clusterTiling = VPU.NCE.ClusterTiling (%input as %arg2: !in_CMX_tensor, %in_weights as %arg3: !weight_CMX_tensor, %in_weights_table as %arg4: !weight_table_CMX_tensor) -> !OTensorDistributed {
      %in = builtin.unrealized_conversion_cast %arg2 : !in_CMX_tensor to memref<1x4x224x224xf16, #NHWC, @CMX_NN>
      %weights = builtin.unrealized_conversion_cast %arg3 : !weight_CMX_tensor to memref<64x1x1x160xf16, #NHWC, @CMX_NN>
      %weight_table = builtin.unrealized_conversion_cast %arg4 : !weight_table_CMX_tensor to memref<64x1x1x4xsi32, @CMX_NN>
      %w_shape_cast = VPUIP.ShapeCast {shape = [64, 16, 7, 7]} inputs(%weights : memref<64x1x1x160xf16, #NHWC, @CMX_NN>) -> memref<64x16x7x7xf16, #NHWC, @CMX_NN>
      %out_alloc = memref.alloc() : memref<1x64x112x112xf16, #NHWC, @CMX_NN>
      %in_shape_cast = VPUIP.ShapeCast {shape = [1, 16, 224, 224]} inputs(%in : memref<1x4x224x224xf16, #NHWC, @CMX_NN>) -> memref<1x16x224x224xf16, #NHWC, @CMX_NN>
      %27 = VPUIP.NCEClusterTask {
            cm_sp_pattern = 15 : i64, input_channels_compression,
            kernel_padding = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64},
            kernel_size = [7, 7], kernel_strides = [2, 2],
            minimumHardwareExecutionCost = 229838 : i64, task_type = "CONV"}
                input(%in_shape_cast : memref<1x16x224x224xf16, #NHWC, @CMX_NN>)
                weights(%w_shape_cast : memref<64x16x7x7xf16, #NHWC, @CMX_NN>)
                weight_table(%weight_table : memref<64x1x1x4xsi32, @CMX_NN>)
                parent_input(%in_shape_cast : memref<1x16x224x224xf16, #NHWC, @CMX_NN>)
                parent_output(%out_alloc : memref<1x64x112x112xf16, #NHWC, @CMX_NN>)
                outputs(%out_alloc : memref<1x64x112x112xf16, #NHWC, @CMX_NN>)
                -> memref<1x64x112x112xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}}
        DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [111, 111, 63], outStart = [0, 56, 0], pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 0 : i64}}
      } PPE : {
        PPETask "NOOP" {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
      }
      %28 = builtin.unrealized_conversion_cast %27 : memref<1x64x112x112xf16, #NHWC, @CMX_NN> to tensor<1x64x112x112xf16, {mem_space = @CMX_NN, order = #NHWC}>
      VPU.Yield %28
    }

    return %clusterTiling : !OTensorDistributed
}

// CHECK:  [[ARG0:%.+]] = builtin.unrealized_conversion_cast %arg2 : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> to memref<64x1x1x4xsi32, @CMX_NN>
// CHECK:  [[ARG1:%.+]] = builtin.unrealized_conversion_cast %arg1 : tensor<64x1x1x160xf16, {mem_space = @CMX_NN, order = #NHWC}> to memref<64x1x1x160xf16, #NHWC, @CMX_NN>
// CHECK:  [[W_SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [64, 16, 7, 7]} inputs([[ARG1]] : memref<64x1x1x160xf16, #NHWC, @CMX_NN>) -> memref<64x16x7x7xf16, #NHWC, @CMX_NN>
// CHECK:  [[ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:  [[ARG2:%.+]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}> to memref<1x4x224x224xf16, #NHWC, @CMX_NN>
// CHECK:  [[IN_SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 224, 224]} inputs([[ARG2]] : memref<1x4x224x224xf16, #NHWC, @CMX_NN>) -> memref<1x16x224x224xf16, #NHWC, @CMX_NN>

// CHECK:  [[CLUSTER_TILING:%.+]] = VPUIP.NCEClusterTiling inputs(
// CHECK-SAME:      [[IN_SHAPE_CAST]] as [[INNER_ARG0:[^:]+]]: memref<1x16x224x224xf16, #NHWC, @CMX_NN>,
// CHECK-SAME:      [[W_SHAPE_CAST]] as [[INNER_ARG1:[^:]+]]: memref<64x16x7x7xf16, #NHWC, @CMX_NN>,
// CHECK-SAME:      [[ARG0]] as [[INNER_ARG2:[^:]+]]: memref<64x1x1x4xsi32, @CMX_NN>) outputs([[ALLOC]] as [[INNER_ARG3:[^:]+]]: memref<1x64x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x64x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {

// CHECK:        [[CLUSTER_TASK:%.+]] = VPUIP.NCEClusterTask {cm_sp_pattern = 15 : i64, input_channels_compression,
// CHECK-SAME:                              kernel_padding = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64},
// CHECK-SAME:                              kernel_size = [7, 7], kernel_strides = [2, 2],
// CHECK-SAME:                              minimumHardwareExecutionCost = 229838 : i64, task_type = "CONV"}
// CHECK-SAME:              input([[INNER_ARG0]] : memref<1x16x224x224xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:              weights([[INNER_ARG1]] : memref<64x16x7x7xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:              weight_table([[INNER_ARG2]] : memref<64x1x1x4xsi32, @CMX_NN>)
// CHECK-SAME:              parent_input([[INNER_ARG0]] : memref<1x16x224x224xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:              parent_output([[INNER_ARG3]] : memref<1x64x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:              outputs([[INNER_ARG3]] : memref<1x64x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x64x112x112xf16, #NHWC, @CMX_NN> variants : {
// CHECK:          DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_16x16", outEnd = [111, 55, 63], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}}
// CHECK:          DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_16x16", outEnd = [111, 111, 63], outStart = [0, 56, 0], pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 0 : i64}}
// CHECK:   [[OUT:%.+]] = builtin.unrealized_conversion_cast [[CLUSTER_TILING]] : !VPUIP.DistributedBuffer<1x64x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME:      to !VPU.DistributedTensor<1x64x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:  return [[OUT]] : !VPU.DistributedTensor<1x64x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

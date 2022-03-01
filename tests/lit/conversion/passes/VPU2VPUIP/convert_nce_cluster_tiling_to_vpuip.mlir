// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB compilation-mode=DefaultHW" --convert-nce-cluster-tiling-to-vpuip --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!type_DDR_tensor = type tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!type_DDR_memref = type memref<1x32x16x16xf16, #NHWC, @DDR>
!type_CMX_tensor = type tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!type_CMX_memref = type memref<1x32x16x16xf16, #NHWC, @CMX_NN>

// NCEClusterTiling operation with memref output
// Original operation before IE2IERT lowering:
// func @NCEClusterTilingCopyOpTensorResult(%input0: !type_DDR_tensor) -> !type_CMX_tensor{
//     %tensor_cmx = VPU.NCE.ClusterTiling(%input0 as %arg0: !type_DDR_tensor) -> !type_CMX_tensor {
//         %0 = IE.Copy(%arg0) { out_mem_space = @CMX_NN } : !type_DDR_tensor -> !type_CMX_tensor
//         VPU.Yield %0
//     }
//     return %tensor_cmx : !type_CMX_tensor
// }

// CHECK-LABEL: @NCEClusterTilingCopyOpTensorResult
func @NCEClusterTilingCopyOpTensorResult(%input0: !type_DDR_memref) -> !type_CMX_memref{

    %input_DDR_tensor = builtin.unrealized_conversion_cast %input0 : !type_DDR_memref
        to !type_DDR_tensor

    %tensor_cmx = VPU.NCE.ClusterTiling(%input_DDR_tensor as %arg0: !type_DDR_tensor) -> !type_CMX_tensor {
        %input_DDR_memref = builtin.unrealized_conversion_cast %arg0: !type_DDR_tensor
            to !type_DDR_memref
        %0 = memref.alloc() : !type_CMX_memref
        %1 = IERT.Copy inputs(%input_DDR_memref : !type_DDR_memref) outputs(%0 : !type_CMX_memref) -> !type_CMX_memref
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
// CHECK:       [[COPY_RES:%.+]] = IERT.Copy
// CHECK-SAME:       inputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, @DDR>)
// CHECK-SAME:      outputs([[ARG2]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!typeCmxDistributed = type !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = OVERLAPPED,
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = {bottom = 1, left = 1, right = 1, top = 1},
    strides = [1, 1],
    num_clusters = 4
}>

!type_DDR_tensor = type tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!type_DDR_memref = type memref<1x32x16x16xf16, #NHWC, @DDR>
!type_CMX_tensor = type tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!type_CMX_memref = type memref<1x32x16x16xf16, #NHWC, @CMX_NN>

// NCEClusterTiling operation with distributed type of output
// Original operation before IE2IERT lowering
// func @NCEClusterTilingCopyOpDistributedResult(%input0: !type_DDR_tensor) -> !type_CMX_tensor{
//     %tensor_distributed_cmx = VPU.NCE.ClusterTiling(%input0 as %arg0: !type_DDR_tensor) -> !typeCmxDistributed {
//         %0 = IE.Copy(%arg0) { out_mem_space = @CMX_NN } : !type_DDR_tensor -> !type_CMX_tensor
//         VPU.Yield %0
//     }
//    %tensor_cmx = builtin.unrealized_conversion_cast %tensor_distributed_cmx : !typeCmxDistributed
//         to !type_CMX_tensor
//     return %tensor_cmx : !type_CMX_tensor
// }
// CHECK-LABEL: @NCEClusterTilingCopyOpDistributedResult
func @NCEClusterTilingCopyOpDistributedResult(%input0: !type_DDR_memref) -> !type_CMX_memref{

    %input_DDR_tensor = builtin.unrealized_conversion_cast %input0 : !type_DDR_memref
        to !type_DDR_tensor

    %tensor_distributed_cmx = VPU.NCE.ClusterTiling(%input_DDR_tensor as %arg0: !type_DDR_tensor) -> !typeCmxDistributed {
        %input_DDR_memref = builtin.unrealized_conversion_cast %arg0: !type_DDR_tensor
            to !type_DDR_memref
        %0 = memref.alloc() : !type_CMX_memref
        %1 = IERT.Copy inputs(%input_DDR_memref : !type_DDR_memref) outputs(%0 : !type_CMX_memref) -> !type_CMX_memref
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
// CHECK:       [[COPY_RES:%.+]] = IERT.Copy
// CHECK-SAME:       inputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, @DDR>)
// CHECK-SAME:      outputs([[ARG2]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!typeCmxDistributed = type !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = OVERLAPPED,
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = {bottom = 1, left = 1, right = 1, top = 1},
    strides = [1, 1],
    num_clusters = 4
}>

!type_DDR_tensor = type tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!type_DDR_memref = type memref<1x32x16x16xf16, #NHWC, @DDR>
!type_CMX_tensor = type tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!type_CMX_memref = type memref<1x32x16x16xf16, #NHWC, @CMX_NN>

// 2 NCEClusterTiling operations with distributed type passed in between
// Original operation before IE2IERT lowering
// func @NCEClusterTilingDistributedCopy2CopyOp(%input0: !type_DDR_tensor) -> !type_DDR_tensor {
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
func @NCEClusterTilingDistributedCopy2CopyOp(%arg0: !type_DDR_memref, %arg1: !type_DDR_memref) -> !type_DDR_memref {
    %0 = builtin.unrealized_conversion_cast %arg0 : !type_DDR_memref to !type_DDR_tensor
    %1 = VPU.NCE.ClusterTiling (%0 as %arg2: !type_DDR_tensor) -> !typeCmxDistributed {
        %5 = builtin.unrealized_conversion_cast %arg2 : !type_DDR_tensor to !type_DDR_memref
        %6 = memref.alloc() : !type_CMX_memref
        %7 = IERT.Copy inputs(%5 : !type_DDR_memref) outputs(%6 : !type_CMX_memref) -> !type_CMX_memref
        %8 = builtin.unrealized_conversion_cast %7 : !type_CMX_memref to !type_CMX_tensor
        VPU.Yield %8
    }
    %2 = VPU.NCE.ClusterTiling (%1 as %arg2: !type_CMX_tensor) -> !type_DDR_tensor {
        %5 = builtin.unrealized_conversion_cast %arg2 : !type_CMX_tensor to !type_CMX_memref
        %6 = memref.alloc() : !type_DDR_memref
        %7 = IERT.Copy inputs(%5 : !type_CMX_memref) outputs(%6 : !type_DDR_memref) -> !type_DDR_memref
        %8 = builtin.unrealized_conversion_cast %7 : !type_DDR_memref to !type_DDR_tensor
        VPU.Yield %8
    }
    %3 = builtin.unrealized_conversion_cast %2 : !type_DDR_tensor to !type_DDR_memref
    %4 = IERT.Copy inputs(%3 : !type_DDR_memref) outputs(%arg1 : !type_DDR_memref) -> !type_DDR_memref
    return %4 : !type_DDR_memref
}

// CHECK:       [[ALLOC0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
// CHECK:       [[NCE_CT_RES0:%.+]] = VPUIP.NCEClusterTiling
// CHECK-SAME:       inputs(%arg0 as [[ARG1:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
// CHECK-SAME:       outputs([[ALLOC0]] as [[ARG2:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:       -> !VPUIP.DistributedBuffer
// CHECK:       [[COPY_RES0:%.+]] = IERT.Copy
// CHECK-SAME:       inputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, @DDR>)
// CHECK-SAME:      outputs([[ARG2]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
// CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x32x16x16xf16, #NHWC, @DDR>
// CHECK:       [[NCE_CT_RES1:%.+]] = VPUIP.NCEClusterTiling
// CHECK-SAME:       inputs([[NCE_CT_RES0]] as [[ARG3:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:       outputs([[ALLOC1]] as [[ARG4:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
// CHECK-SAME:       -> memref<1x32x16x16xf16, #NHWC, @DDR>
// CHECK:       [[COPY_RES1:%.+]] = IERT.Copy
// CHECK-SAME:       inputs([[ARG3]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:      outputs([[ARG4]] : memref<1x32x16x16xf16, #NHWC, @DDR>)

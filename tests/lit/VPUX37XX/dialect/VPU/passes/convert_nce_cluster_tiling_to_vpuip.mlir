//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --convert-nce-cluster-tiling-to-vpuip --canonicalize %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TopKSWClusterTilingSOK
func.func @TopKSWClusterTilingSOK(%arg0: memref<1x1x1x77xf16>, %arg1: memref<1x1x1x1xf16>, %arg2: memref<1x1x1x1xsi32>) -> (memref<1x1x1x1xf16>, memref<1x1x1x1xsi32>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x1x1x77xf16> to tensor<1x1x1x77xf16>
    %1 = VPU.NCE.ClusterTiling (%0 as %arg3: tensor<1x1x1x77xf16>) -> !VPU.DistributedTensor<1x1x1x77xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %9 = builtin.unrealized_conversion_cast %arg3 : tensor<1x1x1x77xf16> to memref<1x1x1x77xf16>
        %10 = memref.alloc() : memref<1x1x1x77xf16, @CMX_NN>
        %11 = VPUIP.Copy inputs(%9 : memref<1x1x1x77xf16>) outputs(%10 : memref<1x1x1x77xf16, @CMX_NN>) -> memref<1x1x1x77xf16, @CMX_NN>
        %12 = builtin.unrealized_conversion_cast %11 : memref<1x1x1x77xf16, @CMX_NN> to tensor<1x1x1x77xf16, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
        VPU.Yield %12 
    }
    %2:2 = VPU.NCE.ClusterTiling (%1 as %arg3: tensor<1x1x1x77xf16, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> (!VPU.DistributedTensor<1x1x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPU.DistributedTensor<1x1x1x1xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) {
        %9 = builtin.unrealized_conversion_cast %arg3 : tensor<1x1x1x77xf16, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> to memref<1x1x1x77xf16, @CMX_NN>
        %10 = memref.alloc() : memref<1x1x1x1xf16, @CMX_NN>
        %11 = memref.alloc() : memref<1x1x1x1xsi32, @CMX_NN>
        %results:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_TopK inputs(%9 as %arg4: memref<1x1x1x77xf16, @CMX_NN>) outputs(%10 as %arg5: memref<1x1x1x1xf16, @CMX_NN>, %11 as %arg6: memref<1x1x1x1xsi32, @CMX_NN>) on tile 0 -> (memref<1x1x1x1xf16, @CMX_NN>, memref<1x1x1x1xsi32, @CMX_NN>){
        VPUIP.SW.Kernel.run {attrs = [0, 0, 0, 1]}(%arg4, %arg5, %arg6) : memref<1x1x1x77xf16, @CMX_NN>, memref<1x1x1x1xf16, @CMX_NN>, memref<1x1x1x1xsi32, @CMX_NN>
        }
        %12 = builtin.unrealized_conversion_cast %results#0 : memref<1x1x1x1xf16, @CMX_NN> to tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
        %13 = builtin.unrealized_conversion_cast %results#1 : memref<1x1x1x1xsi32, @CMX_NN> to tensor<1x1x1x1xsi32, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
        VPU.Yield %12, %13 
    }
    %3 = VPU.NCE.ClusterTiling (%2#0 as %arg3: tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> tensor<1x1x1x1xf16> {
        %9 = builtin.unrealized_conversion_cast %arg3 : tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> to memref<1x1x1x1xf16, @CMX_NN>
        %10 = memref.alloc() : memref<1x1x1x1xf16>
        %11 = VPUIP.Copy inputs(%9 : memref<1x1x1x1xf16, @CMX_NN>) outputs(%10 : memref<1x1x1x1xf16>) -> memref<1x1x1x1xf16>
        %12 = builtin.unrealized_conversion_cast %11 : memref<1x1x1x1xf16> to tensor<1x1x1x1xf16>
        VPU.Yield %12 
    }
    %4 = builtin.unrealized_conversion_cast %3 : tensor<1x1x1x1xf16> to memref<1x1x1x1xf16>
    %5 = VPU.NCE.ClusterTiling (%2#1 as %arg3: tensor<1x1x1x1xsi32, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> tensor<1x1x1x1xsi32> {
        %9 = builtin.unrealized_conversion_cast %arg3 : tensor<1x1x1x1xsi32, {mem_space = @CMX_NN, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> to memref<1x1x1x1xsi32, @CMX_NN>
        %10 = memref.alloc() : memref<1x1x1x1xsi32>
        %11 = VPUIP.Copy inputs(%9 : memref<1x1x1x1xsi32, @CMX_NN>) outputs(%10 : memref<1x1x1x1xsi32>) -> memref<1x1x1x1xsi32>
        %12 = builtin.unrealized_conversion_cast %11 : memref<1x1x1x1xsi32> to tensor<1x1x1x1xsi32>
        VPU.Yield %12 
    }
    %6 = builtin.unrealized_conversion_cast %5 : tensor<1x1x1x1xsi32> to memref<1x1x1x1xsi32>
    %7 = VPUIP.Copy inputs(%4 : memref<1x1x1x1xf16>) outputs(%arg1 : memref<1x1x1x1xf16>) -> memref<1x1x1x1xf16>
    %8 = VPUIP.Copy inputs(%6 : memref<1x1x1x1xsi32>) outputs(%arg2 : memref<1x1x1x1xsi32>) -> memref<1x1x1x1xsi32>
    return %7, %8 : memref<1x1x1x1xf16>, memref<1x1x1x1xsi32>

    // CHECK:       [[OUTPUT_BUFFER:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x1x1x77xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[IN_CLUSTER_TILING:%.+]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg3: memref<1x1x1x77xf16>) outputs([[OUTPUT_BUFFER]] as %arg4: memref<1x1x1x77xf16, @CMX_NN>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x1x1x77xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:           [[INNER_COPY:%.+]] = VPUIP.Copy inputs(%arg3 : memref<1x1x1x77xf16>) outputs(%arg4 : memref<1x1x1x77xf16, @CMX_NN>) -> memref<1x1x1x77xf16, @CMX_NN>
    // CHECK:       }

    // CHECK:       [[BUFFER_DATA_1:%.+]] = VPURT.AllocDistributed
    // CHECK-SMAE:  -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[BUFFER_DATA_2:%.+]] = VPURT.AllocDistributed
    // CHECK-SMAE:  -> !VPUIP.DistributedBuffer<1x1x1x1xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[TOPK_CLUSTER_TILING:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[IN_CLUSTER_TILING]] as %arg3: memref<1x1x1x77xf16, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_DATA_1]] as %arg4: memref<1x1x1x1xf16, @CMX_NN>,
    // CHECK-SAME:      [[BUFFER_DATA_2]] as %arg5: memref<1x1x1x1xsi32, @CMX_NN>)
    // CHECK-SAME:      -> (!VPUIP.DistributedBuffer<1x1x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x1xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) {
    // CHECK:       [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {result_segment_sizes = dense<[2, 0]> : vector<2xi32>} @VPU.SW::@builtin_TopK
    // CHECK-SAME:      inputs(%arg3 as %arg6: memref<1x1x1x77xf16, @CMX_NN>) outputs(%arg4 as %arg7: memref<1x1x1x1xf16, @CMX_NN>, %arg5 as %arg8: memref<1x1x1x1xsi32, @CMX_NN>) on tile 0 -> (memref<1x1x1x1xf16, @CMX_NN>, memref<1x1x1x1xsi32, @CMX_NN>){
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [0, 0, 0, 1]}(%arg6, %arg7, %arg8) : memref<1x1x1x77xf16, @CMX_NN>, memref<1x1x1x1xf16, @CMX_NN>, memref<1x1x1x1xsi32, @CMX_NN>
    // CHECK:           }
    // CHECK:       }

    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc() : memref<1x1x1x1xf16>
    // CHECK:       [[OUTPUT_CLUSTER_TILING:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:  inputs([[TOPK_CLUSTER_TILING]]#0 as %arg3: memref<1x1x1x1xf16, @CMX_NN>) outputs([[ALLOC_1]] as %arg4: memref<1x1x1x1xf16>) -> memref<1x1x1x1xf16> {
    // CHECK:           [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs(%arg3 : memref<1x1x1x1xf16, @CMX_NN>) outputs(%arg4 : memref<1x1x1x1xf16>) -> memref<1x1x1x1xf16>
    // CHECK:       }

    // CHECK:       [[ALLOC_2:%.+]] = memref.alloc() : memref<1x1x1x1xsi32>
    // CHECK:       [[TARGET_CLUSTER_TILING:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:  inputs([[TOPK_CLUSTER_TILING]]#1 as %arg3: memref<1x1x1x1xsi32, @CMX_NN>) outputs([[ALLOC_2]] as %arg4: memref<1x1x1x1xsi32>) -> memref<1x1x1x1xsi32> {
    // CHECK:           [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs(%arg3 : memref<1x1x1x1xsi32, @CMX_NN>) outputs(%arg4 : memref<1x1x1x1xsi32>) -> memref<1x1x1x1xsi32>
    // CHECK:       }

    // CHECK:       [[OUTPUT_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[OUTPUT_CLUSTER_TILING]] : memref<1x1x1x1xf16>) outputs(%arg1 : memref<1x1x1x1xf16>) -> memref<1x1x1x1xf16>

    // CHECK:       [[OUTPUT_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[TARGET_CLUSTER_TILING]] : memref<1x1x1x1xsi32>) outputs(%arg2 : memref<1x1x1x1xsi32>) -> memref<1x1x1x1xsi32>
    
    // CHECK:      return [[OUTPUT_1]], [[OUTPUT_2]] : memref<1x1x1x1xf16>, memref<1x1x1x1xsi32>
}

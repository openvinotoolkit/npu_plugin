//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpuip %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMax
module @SoftMax attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
    // CHECK-DAG: {{  }}module @UsedMemory
    // CHECK-DAG: {{    }}IE.MemoryResource {{[0-9]+}} bytes of @DDR

    // CHECK-DAG: {{  }}IE.TileResource
    // CHECK-DAG: {{    }}builtin.module @UsedMemory
    // CHECK-DAG: {{      }}IE.MemoryResource {{[0-9]+}} bytes of @CMX_NN

    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "singleShaveSoftmax.cpp", VPU.kernel_entry = "singleShaveSoftmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    } outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x1000xf16, @DDR>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR> {
    func.func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
        %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1000xf16>) -> memref<1x1x1x1000xf16>
        %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg2: memref<1x1x1x1000xf16>) outputs(%1 as %arg3: memref<1x1x1x1000xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
            %8 = VPUIP.Copy inputs(%arg2 : memref<1x1x1x1000xf16>) outputs(%arg3 : memref<1x1x1x1000xf16, @CMX_NN>) -> memref<1x1x1x1000xf16, @CMX_NN>
        }
        %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        %4 = VPUIP.NCEClusterTiling inputs(%2 as %arg2: memref<1x1x1x1000xf16, @CMX_NN>) outputs(%3 as %arg3: memref<1x1x1x1000xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
            %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_SoftMax inputs(%arg2 as %arg4: memref<1x1x1x1000xf16, @CMX_NN>) outputs(%arg3 as %arg5: memref<1x1x1x1000xf16, @CMX_NN>) on tile 0 -> memref<1x1x1x1000xf16, @CMX_NN>{
                VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg4, %arg5) : memref<1x1x1x1000xf16, @CMX_NN>, memref<1x1x1x1000xf16, @CMX_NN>
            }
        }
        %alloc = memref.alloc() : memref<1x1x1x1000xf16>
        %5 = VPUIP.NCEClusterTiling inputs(%4 as %arg2: memref<1x1x1x1000xf16, @CMX_NN>) outputs(%alloc as %arg3: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
            %8 = VPUIP.Copy inputs(%arg2 : memref<1x1x1x1000xf16, @CMX_NN>) outputs(%arg3 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
        }
        %6 = VPUIP.GenericReshape inputs(%5 : memref<1x1x1x1000xf16>) -> memref<1x1000xf16>
        %7 = VPUIP.Copy inputs(%6 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
        return %7 : memref<1x1000xf16>

        // CHECK-DAG:   [[BAR0:%.+]] = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
        // CHECK-DAG:   [[BAR1:%.+]] = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
        // CHECK-DAG:   [[BAR2:%.+]] = VPURT.ConfigureBarrier<2> {isFinalBarrier} -> !VPURT.Barrier
        // CHECK-DAG:   [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1000xf16, @DDR>
        // CHECK-DAG:   [[BUFF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        // CHECK-DAG:   [[BUFF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
        // CHECK-DAG:   [[DISTR_BUFF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        // CHECK-DAG:   [[BUFF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        // CHECK-DAG:   [[BUFF3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
        // CHECK-DAG:   [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x1000xf16, @DDR>
        // CHECK-DAG:   [[BUFF4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000xf16, [@CMX_NN, 0]>

        // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x1x1x1000xf16, @DDR>)
        // CHECK-SAME:              outputs([[DISTR_BUFF]] : !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

        // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_SoftMax
        // CHECK-SAME:              inputs([[BUFF0]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
        // CHECK-SAME:              outputs([[BUFF2]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0

        // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0>} @VPU.SW::@builtin_SoftMax
        // CHECK-SAME:              inputs([[BUFF1]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 1]>)
        // CHECK-SAME:              outputs([[BUFF3]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 1]>) on tile 1

        // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[BUFF4]] : memref<1x1000xf16, [@CMX_NN, 0]>) outputs([[OUT]] : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>

        // CHECK: return [[ARG1]] : memref<1x1000xf16, @DDR>
    }
}

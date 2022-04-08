//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: env IE_VPUX_LOG_FILTER="dump-statistics-of-task-ops" vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --dump-statistics-of-task-ops -o /dev/null %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = type !quant.uniform<u8:f16, 1.000000e-01>

module @DumpOpsStatistics attributes {VPU.compilationMode = "DefaultHW"} {

    func @WeightsCompressionStatistics() -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]> {
        %cst_0 = const.Declare memref<1x512x3x3x!qElemType> = dense<1> : tensor<1x512x3x3xui8>, [#const.QuantCast<!qElemType>]
        %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
        VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 1 : i64, isTrailingSWLayer = false} {
            %1 = VPUIP.NNDMA {port = 0 : i64, set_crit = false, set_ord = true}
                inputs(%cst_0 : memref<1x512x3x3x!qElemType>)
                outputs(%0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>)
                -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
        }

        %cst_1 = const.Declare memref<15200x1x1x1xui8> = dense<1> : tensor<15200x1x1x1xui8>
        %2 = VPURT.DeclareBuffer "CMX_NN" <1605632> -> !VPUIP.DistributedBuffer<50176x1x1x1xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        VPURT.Task attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64, isTrailingSWLayer = false} {
            %4 = VPUIP.CompressedDMAOp {port = 0 : i64} inputs(%cst_1 : memref<15200x1x1x1xui8>) outputs(%2 : !VPUIP.DistributedBuffer<50176x1x1x1xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<50176x1x1x1xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        }

        %cst_2 = const.Declare memref<1408x1x1x1xui8> = dense<1> : tensor<1408x1x1x1xui8>
        %3 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
        VPURT.Task attributes {cycleBegin = 2 : i64, cycleEnd = 3 : i64, isTrailingSWLayer = false} {
            %4 = VPUIP.CompressedDMAOp {port = 0 : i64} inputs(%cst_2 : memref<1408x1x1x1xui8>) outputs(%3 : memref<4608x1x1x1xui8, [@CMX_NN, 0]>) -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
        }
        return %0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
    } // func

    // CHECK: VPUIP tasks statistics:
    // CHECK: VPUIP Tasks - 3 ops
    // CHECK:   VPUIP.NNDMA - 1 ops
    // CHECK:     DDR2CMX - 1 ops
    // CHECK:   VPUIP.CompressedDMAOp - 2 ops
    // CHECK:     DDR2CMX - 2 ops

    // CHECK: Weights compression statistics:
    // CHECK: Constants size before compression: 59392 bytes
    // CHECK: Constants size after compression: 21216 bytes
    // CHECK: Constants that were compressed: 16608 bytes (78.28% of total)
    // CHECK: Constants that couldn't be compressed: 4608 bytes (21.72% of total)
    // CHECK: Compression rate of compressed constants: 30.32%
    // CHECK: Total compression rate: 35.72%
}

// -----

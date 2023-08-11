//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --export-VPUIP -o %t %data_path_37XX%/profiling_pll_32.mlir.txt
// RUN: prof_parser -b %t -p %data_path_37XX%/profiling-0-37XX-hw-pll-32.bin -f debug | FileCheck %s

// CHECK: Global offset Section offset    Engine                                                                                          Layer name    IDU dur IDU tstamp SWE ID Res    ODU dur ODU tstamp    Res
// CHECK:             0              0   HWP DPU 2.7                                                             conv1/WithoutBiases/cluster_0/variant_0       1ca5      18606      0   0       20f1      18940      0
// CHECK:            10             10   HWP DPU 2.7                                                                           pool1/cluster_0/variant_0       1a66      1a220      0   0       2229      1a7f3      0
// CHECK:            20             20   HWP DPU 2.7                                                                           pool1/cluster_0/variant_1        ceb      1b3c2      0   0       1123      1b6ed      0
// CHECK:            30             30   HWP DPU 2.7                                                             conv1/WithoutBiases/cluster_1/variant_0       223b      18a07      1   0       26a0      18d53      0
// CHECK:            40             40   HWP DPU 2.7                                                                           pool1/cluster_1/variant_0       16ed      1a21f      1   0       1e82      1a7cf      0
// CHECK:            50             50   HWP DPU 2.7                                                                           pool1/cluster_1/variant_1        b43      1b22a      1   0        f4d      1b532      0
// CHECK: Global offset Section offset    Engine                                                                                          Layer name              Begin   Duration      Stall
// CHECK:            60              0           ACT                                                                conv1/WithoutBiases/cluster_0/tile_0           186f418e        3da          0
// CHECK:            70             10           ACT                                                                             output/cluster_0/tile_0           186f50a6        519          0
// CHECK: Global offset Section offset    Engine                                                                                          Layer name       Begin tstamp         End tstamp
// CHECK:            80              0       DMA 2.7                                                                                 conv1/WithoutBiases           186f3fbf           186f401d
// CHECK:            90             10       DMA 2.7                                                                                 conv1/WithoutBiases           186f45ac           186f45e5
// CHECK:            a0             20       DMA 2.7                                                            conv1/WithoutBiases?_unrolled_permuteDMA           186f45f6           186f4c0b
// CHECK:            b0             30       DMA 2.7                                                     conv1/WithoutBiases?_broadcast_copy_to_CMX[0,1]           186f4c19           186f4c31
// CHECK:            c0             40       DMA 2.7                                                                                               pool1           186f4fbd           186f507b
// CHECK:            d0             50       DMA 2.7                                                     conv1/WithoutBiases?_broadcast_copy_to_CMX[0,1]           186f405c           186f4078
// CHECK:            e0             60       DMA 2.7                                                               conv1/WithoutBiases?_expand_copy_1_13           186f4082           186f4187
// CHECK:            f0             70       DMA 2.7                                                            conv1/WithoutBiases?_unrolled_permuteDMA           186f45ef           186f4ba6
// CHECK:           100             80       DMA 2.7                                       pool1?_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]           186f4bb0           186f4c12
// CHECK:           110             90       DMA 2.7                                                                                               pool1           186f4fb6           186f5094
// CHECK:           120             a0       DMA 2.7                                                                                              output           186f55d8           186f56f4
// CHECK: Global offset                   Engine        PLL Value   WRKPNT CFGID
// CHECK:           130                WORKPOINT               20            202
// CHECK:           134                WORKPOINT               20            202
// CHECK: Engine     Entry size     Number of tasks         Offset    Buffer size
// CHECK:   DPU             10                   6              0             60
// CHECK:    SW             10                   2             60             20
// CHECK:   DMA             10                   b             80             b0
// CHECK: Expected profiling buffer size = 138

//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=VPUX37XX --export-VPUIP -o %t %data_path_37XX%/profiling.mlir.txt
// RUN: prof_parser -b %t -p %data_path_37XX%/profiling-0-37XX.bin -f debug | FileCheck %s

//CHECK:  Global offset Section offset        Engine                                                                                          Layer name    IDU dur IDU tstamp SWE ID Res    ODU dur ODU tstamp    Res
//CHECK:              0              0   HWP DPU 2.7                                               conv1/WithoutBiases?t_Convolution/cluster_0/variant_0        25d       e2e4      0   0        361       e3a8      0
//CHECK:             10             10   HWP DPU 2.7                      conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_0       1e11      11305      0   0       2245      1162d      0
//CHECK:             20             20   HWP DPU 2.7                                                                 pool1?t_MaxPool/cluster_0/variant_0       1a87      12eef      0   0       224c      134c3      0
//CHECK:             30             30   HWP DPU 2.7                                                                 pool1?t_MaxPool/cluster_0/variant_1        cfc      14098      0   0       1138      143c6      0
//CHECK:             40             40   HWP DPU 2.7                                               conv1/WithoutBiases?t_Convolution/cluster_1/variant_0        263       e341      1   0        368       e405      0
//CHECK:             50             50   HWP DPU 2.7                      conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_0       2350      116c4      1   0       2791      119f6      0
//CHECK:             60             60   HWP DPU 2.7                                                                 pool1?t_MaxPool/cluster_1/variant_0       170b      12dea      1   0       1ea0      1339a      0
//CHECK:             70             70   HWP DPU 2.7                                                                 pool1?t_MaxPool/cluster_1/variant_1        b65      13e0e      1   0        f6f      14117      0
//CHECK:  Global offset Section offset        Engine                                                                                          Layer name              Begin   Duration      Stall
//CHECK:             80              0           ACT                                                  conv1/WithoutBiases?t_Convolution/cluster_0/tile_0           18fae404        237        4d7
//CHECK:             90             10           ACT                                                  conv1/WithoutBiases?t_Convolution/cluster_0/tile_1           18fae409        1f3        446
//CHECK:             a0             20           ACT                                                                             output/cluster_0/tile_0           18faec00        135        696
//CHECK:             b0             30           ACT                                                                             output/cluster_0/tile_1           18faec05        11c        687
//CHECK:             e0             60           ACT                                                                             output/cluster_1/tile_0           18faec0a        130        790
//CHECK:             f0             70           ACT                                                                             output/cluster_1/tile_1           18faebfb        12b        5d3
//CHECK:  Global offset Section offset        Engine                                                                                          Layer name       Begin tstamp         End tstamp
//CHECK:            100              0       DMA 2.7                                                                   conv1/WithoutBiases?t_Convolution           18fae2b4           18fae301
//CHECK:            110             10       DMA 2.7                                                                   conv1/WithoutBiases?t_Convolution           18fae64f           18fae67c
//CHECK:            120             20       DMA 2.7                                                                   conv1/WithoutBiases?t_Convolution           18fae685           18fae6ac
//CHECK:            130             30       DMA 2.7conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]           18fae6b8           18fae6d1
//CHECK:            140             40       DMA 2.7                                                                   conv1/WithoutBiases?t_Convolution           18fae6e7           18fae74a
//CHECK:            150             50       DMA 2.7                                          conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]           18fae759           18fae7be
//CHECK:            160             60       DMA 2.7                             pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]           18fae7ca           18fae7e7
//CHECK:            170             70       DMA 2.7                                                                                     pool1?t_MaxPool           18faeab0           18faeb05
//CHECK:            180             80       DMA 2.7                                                                                     pool1?t_MaxPool           18faeb11           18faeb5e
//CHECK:            190             90       DMA 2.7                                                                                     pool1?t_MaxPool           18faeb6d           18faeba4
//CHECK:            1a0             a0       DMA 2.7                                                                                     pool1?t_MaxPool           18faebb0           18faebec
//CHECK:            1b0             b0       DMA 2.7                                                                                              output           18faed54           18faeda8
//CHECK:            1c0             c0       DMA 2.7                                                                                              output           18faedb4           18faedf6
//CHECK:            1d0             d0       DMA 2.7                                                  conv1/WithoutBiases?t_Convolution/_expand_copy_3_2           18fae355           18fae391
//CHECK:            1e0             e0       DMA 2.7                                                                   conv1/WithoutBiases?t_Convolution           18fae68b           18fae6b2
//CHECK:            1f0             f0       DMA 2.7                                                                   conv1/WithoutBiases?t_Convolution           18fae6e1           18fae750
//CHECK:            200            100       DMA 2.7                                          conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]           18fae75f           18fae7c4
//CHECK:            210            110       DMA 2.7                                                                                     pool1?t_MaxPool           18faeaaa           18faeaff
//CHECK:            220            120       DMA 2.7                                                                                     pool1?t_MaxPool           18faeb0b           18faeb58
//CHECK:            230            130       DMA 2.7                                                                                     pool1?t_MaxPool           18faeb67           18faeb9e
//CHECK:            240            140       DMA 2.7                                                                                     pool1?t_MaxPool           18faebaa           18faebe6
//CHECK:            250            150       DMA 2.7                                                                                              output           18faed4e           18faedae
//CHECK:            260            160       DMA 2.7                                                                                              output           18faedbc           18faee08
//CHECK:  Global offset                   Engine        PLL Value   WRKPNT CFGID
//CHECK:            280                WORKPOINT               27            202
//CHECK:            284                WORKPOINT               27            202
//CHECK: Engine     Entry size     Number of tasks         Offset    Buffer size
//CHECK:    DPU             10                   8              0             80
//CHECK:     SW             10                   8             80             80
//CHECK:    DMA             10                  17            100            170
//CHECK: Expected profiling buffer size = 2c0
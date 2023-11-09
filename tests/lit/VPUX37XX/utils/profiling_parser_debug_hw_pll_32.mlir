//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=VPUX37XX --export-VPUIP %data_path_37XX%/profiling.mlir.txt -o %t
// RUN: prof_parser -b %t -p %data_path_37XX%/profiling-0-37XX-hw-pll-32.bin -f debug | FileCheck %s

//CHECK:  Global offset Section offset        Engine                                                                                          Layer name    IDU dur IDU tstamp SWE ID Res    ODU dur ODU tstamp    Res
//CHECK:              0              0   HWP DPU 2.7                                               conv1/WithoutBiases?t_Convolution/cluster_0/variant_0        263       dec5      0   0        367       df89      0
//CHECK:             10             10   HWP DPU 2.7                      conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_0       1dfb      10d34      0   0       222a      11058      0
//CHECK:             20             20   HWP DPU 2.7                                                                 pool1?t_MaxPool/cluster_0/variant_0       1a7b      12962      0   0       2244      12f39      0
//CHECK:             30             30   HWP DPU 2.7                                                                 pool1?t_MaxPool/cluster_0/variant_1        d0a      13b0f      0   0       1146      13e3d      0
//CHECK:             40             40   HWP DPU 2.7                                               conv1/WithoutBiases?t_Convolution/cluster_1/variant_0        26d       dee6      1   0        373       dfab      0
//CHECK:             50             50   HWP DPU 2.7                      conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_0       237a      11169      1   0       27ad      11490      0
//CHECK:             60             60   HWP DPU 2.7                                                                 pool1?t_MaxPool/cluster_1/variant_0       1725      1289f      1   0       1eba      12e4f      0
//CHECK:             70             70   HWP DPU 2.7                                                                 pool1?t_MaxPool/cluster_1/variant_1        b6b      138c6      1   0        f75      13bcf      0
//CHECK:  Global offset Section offset        Engine                                                                                          Layer name              Begin   Duration      Stall
//CHECK:             80              0           ACT                                                  conv1/WithoutBiases?t_Convolution/cluster_0/tile_0           18840482        292        49c
//CHECK:             90             10           ACT                                                  conv1/WithoutBiases?t_Convolution/cluster_0/tile_1           18840487        23b        3ce
//CHECK:             a0             20           ACT                                                                             output/cluster_0/tile_0           18840de4        16e        697
//CHECK:             b0             30           ACT                                                                             output/cluster_0/tile_1           18840ddf        158        5cd
//CHECK:             e0             60           ACT                                                                             output/cluster_1/tile_0           18840de9        198        aa7
//CHECK:             f0             70           ACT                                                                             output/cluster_1/tile_1           18840dda        17d        8c5
//CHECK:  Global offset Section offset        Engine                                                                                          Layer name       Begin tstamp         End tstamp
//CHECK:            100              0       DMA 2.7                                                                   conv1/WithoutBiases?t_Convolution           188402f1           1884034c
//CHECK:            110             10       DMA 2.7                                                                   conv1/WithoutBiases?t_Convolution           1884072c           18840762
//CHECK:            120             20       DMA 2.7                                                                   conv1/WithoutBiases?t_Convolution           1884076c           18840792
//CHECK:            130             30       DMA 2.7conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]           188407a3           188407bf
//CHECK:            140             40       DMA 2.7                                                                   conv1/WithoutBiases?t_Convolution           188407db           1884084a
//CHECK:            150             50       DMA 2.7                                          conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]           1884085b           188408d8
//CHECK:            160             60       DMA 2.7                             pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]           188408e2           188408f7
//CHECK:            170             70       DMA 2.7                                                                                     pool1?t_MaxPool           18840c69           18840cbc
//CHECK:            180             80       DMA 2.7                                                                                     pool1?t_MaxPool           18840cca           18840d19
//CHECK:            190             90       DMA 2.7                                                                                     pool1?t_MaxPool           18840d39           18840d70
//CHECK:            1a0             a0       DMA 2.7                                                                                     pool1?t_MaxPool           18840d7a           18840db3
//CHECK:            1b0             b0       DMA 2.7                                                                                              output           18840f99           18840ff7
//CHECK:            1c0             c0       DMA 2.7                                                                                              output           18841005           1884104f
//CHECK:            1d0             d0       DMA 2.7                                                  conv1/WithoutBiases?t_Convolution/_expand_copy_3_2           188403b5           188403fb
//CHECK:            1e0             e0       DMA 2.7                                                                   conv1/WithoutBiases?t_Convolution           18840773           1884079c
//CHECK:            1f0             f0       DMA 2.7                                                                   conv1/WithoutBiases?t_Convolution           188407d4           18840843
//CHECK:            200            100       DMA 2.7                                          conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]           18840854           188408c7
//CHECK:            210            110       DMA 2.7                                                                                     pool1?t_MaxPool           18840c62           18840cc3
//CHECK:            220            120       DMA 2.7                                                                                     pool1?t_MaxPool           18840cd4           18840d2f
//CHECK:            230            130       DMA 2.7                                                                                     pool1?t_MaxPool           18840d40           18840d81
//CHECK:            240            140       DMA 2.7                                                                                     pool1?t_MaxPool           18840d8b           18840dc8
//CHECK:            250            150       DMA 2.7                                                                                              output           18840fa0           18840ffe
//CHECK:            260            160       DMA 2.7                                                                                              output           1884100f           18841064
//CHECK:  Global offset                   Engine        PLL Value   WRKPNT CFGID
//CHECK:            280                WORKPOINT               20            202
//CHECK:            284                WORKPOINT               20            202
//CHECK: Engine     Entry size     Number of tasks         Offset    Buffer size
//CHECK:    DPU             10                   8              0             80
//CHECK:     SW             10                   8             80             80
//CHECK:    DMA             10                  17            100            170
//CHECK: Expected profiling buffer size = 2c0

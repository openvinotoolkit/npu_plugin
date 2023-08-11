//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
//
// RUN: vpux-translate --export-VPUIP -o %t %data_path_30XX%/profiling.mlir.txt
// RUN: prof_parser -b %t -p %data_path_30XX%/profiling-0-30XX.bin -f text | FileCheck %s
//

// CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution                            Time(us): 47.91         Start(us): 277.42
// CHECK: Task(DPU): pool1?t_Convolution                                          Time(us): 8.46          Start(us): 325.71
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_expand_copy_1_13          Time(us): 37.47         Start(us): 0.00
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                            Time(us): 7.77          Start(us): 67.26
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                            Time(us): 9.15          Start(us): 235.95
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                            Time(us): 9.68          Start(us): 245.24
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                            Time(us): 10.52         Start(us): 255.07
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                            Time(us): 6.83          Start(us): 265.74
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]    Time(us): 4.70          Start(us): 272.72
// CHECK: Task(DMA): pool1?t_Convolution/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]  Time(us): 0.68          Start(us): 277.59
// CHECK: Task(DMA): pool1?t_Convolution                                          Time(us): 2.25          Start(us): 335.58
// CHECK: Task(DMA): pool1?t_Convolution                                          Time(us): 2.08          Start(us): 338.09
// CHECK: Task(DMA): pool1?t_Convolution                                          Time(us): 2.20          Start(us): 340.47
// CHECK: Task(DMA): pool1?t_Convolution                                          Time(us): 1.45          Start(us): 342.83
// CHECK: Task(DMA): output                                                       Time(us): 20.37         Start(us): 344.54
// CHECK: Task(DMA): output?_unrolled_permuteDMA                                  Time(us): 36.39         Start(us): 365.06
// CHECK: Task(DMA): output?_unrolled_permuteDMA                                  Time(us): 36.40         Start(us): 401.61
// CHECK: Task(DMA): output?_unrolled_permuteDMA                                  Time(us): 36.39         Start(us): 438.15
// CHECK: Task(DMA): output?_unrolled_permuteDMA                                  Time(us): 18.98         Start(us): 474.68
// CHECK: Task(DMA): output                                                       Time(us): 7.15          Start(us): 493.84
// CHECK: Task(SW): conv1/WithoutBiases?t_Convolution                             Time(us): 37.70         Cycles:136516(68247)    Start(us): 24.21
// CHECK: Task(SW): conv1/WithoutBiases?t_Convolution                             Time(us): 155.57        Cycles:221810(1155920)  Start(us): 75.03
// CHECK: Task(SW): output                                                        Time(us): 65.08         Cycles:191200(233572)   Start(us): 502.99
// CHECK: Layer: conv1/WithoutBiases                      Type: Convolution          DPU: 47.91    SW: 193.27   DMA: 86.13        Start: 0.00
// CHECK: Layer: pool1                                    Type: Convolution          DPU: 8.46     SW: 0.00     DMA: 8.66         Start: 277.59
// CHECK: Layer: output                                   Type:                      DPU: 0.00     SW: 65.08    DMA: 155.67       Start: 344.54
// CHECK: Total time: 565.18us, Real: 568.07us

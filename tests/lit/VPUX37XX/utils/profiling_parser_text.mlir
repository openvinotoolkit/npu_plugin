//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=VPUX37XX --export-VPUIP -o %t %data_path_37XX%/profiling.mlir.txt
// RUN: prof_parser -b %t -p %data_path_37XX%/profiling-0-37XX.bin -f text | FileCheck %s

//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                                   Time(us): 2.01         Start(us): 0.00    
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_expand_copy_3_2                  Time(us): 1.56         Start(us): 4.19    
//CHECK: Task(SW): conv1/WithoutBiases?t_Convolution                            Time(us): 14.77        Cycles:0(0)     Start(us): 8.75    
//CHECK: Task(SW): conv1/WithoutBiases?t_Convolution/cluster_0                  Time(us): 14.77        Cycles:0(0)     Start(us): 8.75    
//CHECK: Task(SW): conv1/WithoutBiases?t_Convolution/cluster_0/tile_0           Time(us): 14.77        Cycles:0(1239)  Start(us): 8.75    
//CHECK: Task(SW): conv1/WithoutBiases?t_Convolution/cluster_0/tile_1           Time(us): 12.99        Cycles:0(1094)  Start(us): 8.88    
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                                   Time(us): 1.17         Start(us): 24.04   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                                   Time(us): 1.01         Start(us): 25.44   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                                   Time(us): 1.01         Start(us): 25.60   
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution                                   Time(us): 0.76         Start(us): 26.62   
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_0                         Time(us): 0.67         Start(us): 26.62   
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_1                         Time(us): 0.67         Start(us): 26.71   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1] Time(us): 0.65          Start(us): 26.77   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                                   Time(us): 2.89         Start(us): 27.84   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                                   Time(us): 2.58         Start(us): 28.00   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]          Time(us): 2.63         Start(us): 30.96   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]          Time(us): 2.63         Start(us): 31.12   
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]          Time(us): 7.79         Start(us): 33.75   
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1      Time(us): 7.79           Start(us): 33.75   
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0      Time(us): 6.75           Start(us): 33.80   
//CHECK: Task(DMA): pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]   Time(us): 0.75           Start(us): 33.91   
//CHECK: Task(DPU): pool1?t_MaxPool                                                     Time(us): 10.69        Start(us): 41.83   
//CHECK: Task(DPU): pool1?t_MaxPool/cluster_0                                           Time(us): 10.69        Start(us): 41.83   
//CHECK: Task(DPU): pool1?t_MaxPool/cluster_1                                           Time(us): 9.57         Start(us): 42.24   
//CHECK: Task(DMA): pool1?t_MaxPool                                                     Time(us): 2.21         Start(us): 53.07   
//CHECK: Task(DMA): pool1?t_MaxPool                                                     Time(us): 2.21         Start(us): 53.23   
//CHECK: Task(DMA): pool1?t_MaxPool                                                     Time(us): 2.01         Start(us): 55.60   
//CHECK: Task(DMA): pool1?t_MaxPool                                                     Time(us): 2.01         Start(us): 55.76   
//CHECK: Task(DMA): pool1?t_MaxPool                                                     Time(us): 1.43         Start(us): 57.99   
//CHECK: Task(DMA): pool1?t_MaxPool                                                     Time(us): 1.43         Start(us): 58.15   
//CHECK: Task(DMA): pool1?t_MaxPool                                                     Time(us): 1.56         Start(us): 59.74   
//CHECK: Task(DMA): pool1?t_MaxPool                                                     Time(us): 1.56         Start(us): 59.90   
//CHECK: Task(SW): output                                                       Time(us): 8.31         Cycles:0(0)     Start(us): 61.85   
//CHECK: Task(SW): output/cluster_1                                             Time(us): 8.31         Cycles:0(0)     Start(us): 61.85   
//CHECK: Task(SW): output/cluster_1/tile_1                                      Time(us): 7.79         Cycles:0(1491)  Start(us): 61.85   
//CHECK: Task(SW): output/cluster_0                                             Time(us): 8.05         Cycles:0(0)     Start(us): 61.98   
//CHECK: Task(SW): output/cluster_0/tile_0                                      Time(us): 8.05         Cycles:0(1686)  Start(us): 61.98   
//CHECK: Task(SW): output/cluster_0/tile_1                                      Time(us): 7.39         Cycles:0(1671)  Start(us): 62.11   
//CHECK: Task(SW): output/cluster_1/tile_0                                      Time(us): 7.92         Cycles:0(1936)  Start(us): 62.24   
//CHECK: Task(DMA): output                                                              Time(us): 2.50         Start(us): 70.68   
//CHECK: Task(DMA): output                                                              Time(us): 2.19         Start(us): 70.83   
//CHECK: Task(DMA): output                                                              Time(us): 1.72         Start(us): 73.33   
//CHECK: Task(DMA): output                                                              Time(us): 1.98         Start(us): 73.54   
//CHECK: Layer: conv1/WithoutBiases                      Type: Convolution          DPU: 8.55     SW: 14.77    DMA: 18.15      Start: 0.00
//CHECK: Layer: pool1                                    Type: MaxPool              DPU: 10.69    SW: 0.00     DMA: 15.18      Start: 33.91
//CHECK: Layer: output                                   Type:                      DPU: 0.00     SW: 8.31     DMA: 8.38       Start: 61.85
//CHECK: Total time: 84.03us, Real: 75.52us

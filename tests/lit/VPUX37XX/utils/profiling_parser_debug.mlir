//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --export-VPUIP -o %t %data_path_37XX%/profiling.mlir.txt
// RUN: prof_parser -b %t -p %data_path_37XX%/profiling-0-37XX.bin -f text | FileCheck %s

// CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution                           	Time(us): 7.60    	Start(us): 67.11
// CHECK: Task(DPU): pool1?t_Convolution                                         	Time(us): 10.65   	Start(us): 74.97
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                           	Time(us): 2.13    	Start(us): 0.00
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_broadcast_copy_to_CMX[0,1]	Time(us): 0.57    	Start(us): 3.67
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_expand_copy_1_13         	Time(us): 5.26    	Start(us): 4.48
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                           	Time(us): 1.20    	Start(us): 31.38
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_unrolled_permuteDMA      	Time(us): 33.31   	Start(us): 32.97
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_unrolled_permuteDMA      	Time(us): 31.33   	Start(us): 32.81
// CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_broadcast_copy_to_CMX[0,1]	Time(us): 0.60    	Start(us): 66.51
// CHECK: Task(DMA): pool1?t_Convolution/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]	Time(us): 0.78    	Start(us): 64.38
// CHECK: Task(DMA): pool1?t_Convolution                                         	Time(us): 4.35    	Start(us): 86.48
// CHECK: Task(DMA): pool1?t_Convolution                                         	Time(us): 4.69    	Start(us): 86.33
// CHECK: Task(DMA): output                                                      	Time(us): 6.04    	Start(us): 119.48
// CHECK: Task(SW): conv1/WithoutBiases?t_Convolution                           	Time(us): 20.10   	Cycles:0(0)	Start(us): 9.90
// CHECK: Task(SW): output                                                      	Time(us): 27.50   	Cycles:0(0)	Start(us): 91.46
// CHECK: Layer: conv1/WithoutBiases                      Type: Convolution          DPU: 7.60     SW: 20.10    DMA: 74.40   	Start: 0.00
// CHECK: Layer: pool1                                    Type: Convolution          DPU: 10.65    SW: 0.00     DMA: 9.82    	Start: 64.38
// CHECK: Layer: output                                   Type:                      DPU: 0.00     SW: 27.50    DMA: 6.04    	Start: 91.46
// CHECK: Total time: 156.10us, Real: 125.52us

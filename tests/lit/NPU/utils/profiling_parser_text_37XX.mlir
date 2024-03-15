//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t %data_path_npu%/profiling-37XX.mlir.txt
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-37XX.bin -f text | FileCheck %s
// REQUIRES: arch-VPUX37XX

//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                           	Time(us): 2.01    	Start(us): 0.00    
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_expand_copy_3_2          	Time(us): 1.56    	Start(us): 4.09    
//CHECK: Task(SW): conv1/WithoutBiases?t_Convolution/tile_0/cluster_0          	Time(us): 9.45    	Cycles:0(1351)	Start(us): 9.90    
//CHECK: Task(SW): conv1/WithoutBiases?t_Convolution/tile_1/cluster_0          	Time(us): 7.34    	Cycles:0(1178)	Start(us): 10.03   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution                           	Time(us): 1.17    	Start(us): 19.87   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_cluster_1                	Time(us): 0.86    	Start(us): 21.28   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_cluster_0                	Time(us): 0.88    	Start(us): 21.43   
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_0                 	Time(us): 0.66    	Start(us): 22.35   
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/cluster_1                 	Time(us): 0.67    	Start(us): 22.38   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_fused_constant/_fused_tile	Time(us): 0.70    	Start(us): 22.55   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_cluster_1                	Time(us): 2.76    	Start(us): 23.46   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_cluster_0                	Time(us): 2.76    	Start(us): 23.62   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_cluster_1                	Time(us): 2.60    	Start(us): 26.61   
//CHECK: Task(DMA): conv1/WithoutBiases?t_Convolution/_cluster_0                	Time(us): 2.76    	Start(us): 26.77   
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_0    	Time(us): 32.48   	Start(us): 29.53   
//CHECK: Task(DPU): conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_1    	Time(us): 33.10   	Start(us): 29.59   
//CHECK: Task(DMA): conv2/WithoutBiases?t_Convolution/_fused_constant/_fused_tile	Time(us): 2.32    	Start(us): 29.77   
//CHECK: Task(DPU): relu1?t_Relu/cluster_0                                      	Time(us): 10.07   	Start(us): 63.68   
//CHECK: Task(DPU): relu1?t_Relu/cluster_1                                      	Time(us): 6.71    	Start(us): 63.94   
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_1                 	Time(us): 289.01  	Start(us): 74.00   
//CHECK: Task(DPU): conv2/WithoutBiases?t_Convolution/cluster_0                 	Time(us): 285.98  	Start(us): 74.38   
//CHECK: Task(DPU): relu2?t_Relu/cluster_0                                      	Time(us): 4.35    	Start(us): 363.19  
//CHECK: Task(DPU): relu2?t_Relu/cluster_1                                      	Time(us): 3.54    	Start(us): 363.65  
//CHECK: Task(DMA): relu2?t_Relu/_cluster_1                                     	Time(us): 1.12    	Start(us): 368.83  
//CHECK: Task(DMA): relu2?t_Relu/_cluster_0                                     	Time(us): 0.81    	Start(us): 368.98  
//CHECK: Task(DMA): relu2?t_Relu/_cluster_0                                     	Time(us): 0.62    	Start(us): 370.10  
//CHECK: Task(DMA): relu2?t_Relu/_cluster_1                                     	Time(us): 0.70    	Start(us): 370.31  
//CHECK: Task(DMA): relu2?t_Relu/_cluster_0                                     	Time(us): 0.75    	Start(us): 371.25  
//CHECK: Task(DMA): relu2?t_Relu/_cluster_1                                     	Time(us): 0.75    	Start(us): 371.41  
//CHECK: Task(DMA): relu2?t_Relu/_cluster_0                                     	Time(us): 0.81    	Start(us): 372.32  
//CHECK: Task(DMA): relu2?t_Relu/_cluster_1                                     	Time(us): 0.81    	Start(us): 372.47  
//CHECK: Task(SW): output?t_Output/tile_1/cluster_1                            	Time(us): 4.89    	Cycles:0(822)	Start(us): 373.67  
//CHECK: Task(SW): output?t_Output/tile_0/cluster_1                            	Time(us): 5.03    	Cycles:0(997)	Start(us): 373.80  
//CHECK: Task(SW): output?t_Output/tile_1/cluster_0                            	Time(us): 4.51    	Cycles:0(998)	Start(us): 373.93  
//CHECK: Task(SW): output?t_Output/tile_0/cluster_0                            	Time(us): 4.64    	Cycles:0(1188)	Start(us): 374.06  
//CHECK: Task(DMA): output?t_Output/_cluster_0                                  	Time(us): 0.91    	Start(us): 379.35  
//CHECK: Task(DMA): output?t_Output/_cluster_1                                  	Time(us): 0.91    	Start(us): 379.51  
//CHECK: Task(DMA): output?t_Output/_cluster_0                                  	Time(us): 0.62    	Start(us): 380.57  
//CHECK: Task(DMA): output?t_Output/_cluster_1                                  	Time(us): 0.62    	Start(us): 380.73  
//CHECK: Layer: conv1/WithoutBiases                      Type: Convolution          DPU: 66.91    SW: 16.80    DMA: 18.07   	Start: 0.00
//CHECK: Layer: conv2/WithoutBiases                      Type: Convolution          DPU: 574.98   SW: 0.00     DMA: 2.32    	Start: 29.77
//CHECK: Layer: relu1                                    Type: Relu                 DPU: 16.78    SW: 0.00     DMA: 0.00    	Start: 63.68
//CHECK: Layer: relu2                                    Type: Relu                 DPU: 7.88     SW: 0.00     DMA: 6.38    	Start: 363.19
//CHECK: Layer: output                                   Type: Output               DPU: 0.00     SW: 19.06    DMA: 3.07    	Start: 373.67
//CHECK: Total time: 732.25us, Real: 381.35us

//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t %data_path_npu%/profiling-37XX.mlir.txt
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-37XX.bin -f debug | FileCheck %s
// REQUIRES: arch-VPUX37XX

//CHECK:   Index  Offset        Engine  Buffer ID         Cluster ID      Buffer offset    IDU dur IDU tstamp SWE ID Rvd    ODU dur ODU tstamp    Rvd  Task
//CHECK:       0       0           dpu          0                  0                  0        24f       d838      0   0        35c       d902      0  conv1/WithoutBiases?t_Convolution/cluster_0/variant_0
//CHECK:       1      10           dpu          0                  0                 10       a1d0      16b32      0   0       a4f1      16d8b      0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_0/variant_0
//CHECK:       2      20           dpu          0                  0                 20       1f66      18b75      0   0       225b      18dae      0  relu1?t_Relu/cluster_0/variant_0
//CHECK:       3      30           dpu          0                  0                 30       11f2      198ed      0   0       10c8      19a46      0  relu1?t_Relu/cluster_0/variant_1
//CHECK:       4      40           dpu          0                  0                 40      5a75d      5da29      0   0      5ac38      5ddce      0  conv2/WithoutBiases?t_Convolution/cluster_0/variant_0
//CHECK:       5      50           dpu          0                  0                 50        e47      5f355      0   0       1612      5f92e      0  relu2?t_Relu/cluster_0/variant_0
//CHECK:       6      60           dpu          0                  1                  0        257       d857      1   0        361       d920      0  conv1/WithoutBiases?t_Convolution/cluster_1/variant_0
//CHECK:       7      70           dpu          0                  1                 10       a4f6      16dcd      1   0       a819      17028      0  conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_1/variant_0
//CHECK:       8      80           dpu          0                  1                 20       145f      1842e      1   0       171b      1863c      0  relu1?t_Relu/cluster_1/variant_0
//CHECK:       9      90           dpu          0                  1                 30        c0f      18d3c      1   0        af0      18e73      0  relu1?t_Relu/cluster_1/variant_1
//CHECK:      10      a0           dpu          0                  1                 40      5b68a      5e41c      1   0      5bb9c      5e7eb      0  conv2/WithoutBiases?t_Convolution/cluster_1/variant_0
//CHECK:      11      b0           dpu          0                  1                 50        a67      5f22b      1   0       11f5      5f7d6      0  relu2?t_Relu/cluster_1/variant_0

//CHECK:   Index  Offset        Engine  Buffer ID         Cluster ID      Buffer offset              Begin   Duration      Stall  Task
//CHECK:       0      c0      actshave          0                  0                  0           1aecb607        16b        547  conv1/WithoutBiases?t_Convolution/tile_0/cluster_0
//CHECK:       1      d0      actshave          0                  0                 10           1aecb60c        11a        49a  conv1/WithoutBiases?t_Convolution/tile_1/cluster_0
//CHECK:       2      e0      actshave          0                  0                 20           1aececa7         b2        4a4  output?t_Output/tile_0/cluster_0
//CHECK:       3      f0      actshave          0                  0                 30           1aececa2         ad        3e6  output?t_Output/tile_1/cluster_0
//CHECK:       4     120      actshave          0                  1                 20           1aecec9d         c1        3e5  output?t_Output/tile_0/cluster_1
//CHECK:       5     130      actshave          0                  1                 30           1aecec98         bc        336  output?t_Output/tile_1/cluster_1

//CHECK:   Index  Offset        Engine       Begin tstamp         End tstamp  Task
//CHECK:       0     140           dma           1aecb48b           1aecb4d8  conv1/WithoutBiases?t_Convolution
//CHECK:       1     150           dma           1aecb786           1aecb7b3  conv1/WithoutBiases?t_Convolution
//CHECK:       2     160           dma           1aecb7c2           1aecb7e4  conv1/WithoutBiases?t_Convolution/_cluster_0
//CHECK:       3     170           dma           1aecb7ed           1aecb808  conv1/WithoutBiases?t_Convolution/_fused_constant/_fused_tile
//CHECK:       4     180           dma           1aecb816           1aecb880  conv1/WithoutBiases?t_Convolution/_cluster_0
//CHECK:       5     190           dma           1aecb88f           1aecb8f9  conv1/WithoutBiases?t_Convolution/_cluster_0
//CHECK:       6     1a0           dma           1aecb902           1aecb95b  conv2/WithoutBiases?t_Convolution/_fused_constant/_fused_tile
//CHECK:       7     1b0           dma           1aecebe4           1aecec03  relu2?t_Relu/_cluster_0
//CHECK:       8     1c0           dma           1aecec0f           1aecec27  relu2?t_Relu/_cluster_0
//CHECK:       9     1d0           dma           1aecec3b           1aecec58  relu2?t_Relu/_cluster_0
//CHECK:      10     1e0           dma           1aecec64           1aecec83  relu2?t_Relu/_cluster_0
//CHECK:      11     1f0           dma           1aeced72           1aeced95  output?t_Output/_cluster_0
//CHECK:      12     200           dma           1aeceda1           1aecedb9  output?t_Output/_cluster_0
//CHECK:      13     210           dma           1aecb528           1aecb564  conv1/WithoutBiases?t_Convolution/_expand_copy_3_2
//CHECK:      14     220           dma           1aecb7bc           1aecb7dd  conv1/WithoutBiases?t_Convolution/_cluster_1
//CHECK:      15     230           dma           1aecb810           1aecb87a  conv1/WithoutBiases?t_Convolution/_cluster_1
//CHECK:      16     240           dma           1aecb889           1aecb8ed  conv1/WithoutBiases?t_Convolution/_cluster_1
//CHECK:      17     250           dma           1aecebde           1aecec09  relu2?t_Relu/_cluster_1
//CHECK:      18     260           dma           1aecec17           1aecec32  relu2?t_Relu/_cluster_1
//CHECK:      19     270           dma           1aecec41           1aecec5e  relu2?t_Relu/_cluster_1
//CHECK:      20     280           dma           1aecec6a           1aecec89  relu2?t_Relu/_cluster_1
//CHECK:      21     290           dma           1aeced78           1aeced9b  output?t_Output/_cluster_1
//CHECK:      22     2a0           dma           1aeceda7           1aecedbf  output?t_Output/_cluster_1

//CHECK:   Index  Offset        Engine        PLL Value          CFGID
//CHECK:       0     2c0           pll               27            202
//CHECK:       1     2c4           pll               27            202


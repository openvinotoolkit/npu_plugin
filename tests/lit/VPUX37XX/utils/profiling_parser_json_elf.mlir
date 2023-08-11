//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="vpu-arch=VPUX37XX allow-custom-values=true" --lower-VPUIP-to-ELF %data_path_37XX%/profiling.mlir.txt | vpux-translate --export-ELF -o %t
// RUN: prof_parser -b %t -p %data_path_37XX%/profiling-0-37XX.bin -f json | FileCheck %s

// CHECK: {"traceEvents":[
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"DPU", "ph":"X", "ts":67.109, "dur":7.597, "pid":1, "tid":3},
// CHECK: {"name":"pool1?t_Convolution", "cat":"DPU", "ph":"X", "ts":74.970, "dur":10.647, "pid":1, "tid":3},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"DMA", "ph":"X", "ts":0.000, "dur":2.135, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/_broadcast_copy_to_CMX[0,1]", "cat":"DMA", "ph":"X", "ts":3.672, "dur":0.572, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/_expand_copy_1_13", "cat":"DMA", "ph":"X", "ts":4.479, "dur":5.260, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"DMA", "ph":"X", "ts":31.380, "dur":1.197, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":32.968, "dur":33.307, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":32.812, "dur":31.328, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/_broadcast_copy_to_CMX[0,1]", "cat":"DMA", "ph":"X", "ts":66.510, "dur":0.598, "pid":1, "tid":2},
// CHECK: {"name":"pool1?t_Convolution/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,1]", "cat":"DMA", "ph":"X", "ts":64.375, "dur":0.781, "pid":1, "tid":2},
// CHECK: {"name":"pool1?t_Convolution", "cat":"DMA", "ph":"X", "ts":86.484, "dur":4.348, "pid":1, "tid":2},
// CHECK: {"name":"pool1?t_Convolution", "cat":"DMA", "ph":"X", "ts":86.328, "dur":4.687, "pid":1, "tid":2},
// CHECK: {"name":"output", "cat":"DMA", "ph":"X", "ts":119.479, "dur":6.041, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"SW", "ph":"X", "ts":9.896, "dur":20.104, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
// CHECK: {"name":"output", "cat":"SW", "ph":"X", "ts":91.458, "dur":27.500, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
// CHECK: {"name":"conv1/WithoutBiases", "cat":"Layer", "ph":"X", "ts":0.000, "dur":74.706, "pid":1, "tid":5, "args":{"Layer type": "Convolution"}},
// CHECK: {"name":"pool1", "cat":"Layer", "ph":"X", "ts":64.375, "dur":26.640, "pid":1, "tid":5, "args":{"Layer type": "Convolution"}},
// CHECK: {"name":"output", "cat":"Layer", "ph":"X", "ts":91.458, "dur":34.062, "pid":1, "tid":5, "args":{"Layer type": ""}},
// CHECK: {"name": "process_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name" : "Inference"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 2, "args": {"name" : "DMA"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 3, "args": {"name" : "DPU"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 4, "args": {"name" : "SW"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 5, "args": {"name" : "Layers"}}
// CHECK: ],
// CHECK: "displayTimeUnit": "ns"
// CHECK: }

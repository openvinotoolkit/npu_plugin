//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
//
// RUN: vpux-translate --export-VPUIP -o %t %data_path_30XX%/profiling.mlir.txt
// RUN: prof_parser -b %t -p %data_path_30XX%/profiling-0-30XX.bin -f json | FileCheck %s

// CHECK: {"traceEvents":[
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"DPU", "ph":"X", "ts":277.419, "dur":47.911, "pid":1, "tid":3},
// CHECK: {"name":"pool1?t_Convolution", "cat":"DPU", "ph":"X", "ts":325.707, "dur":8.461, "pid":1, "tid":3},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/_expand_copy_1_13", "cat":"DMA", "ph":"X", "ts":0.000, "dur":37.474, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"DMA", "ph":"X", "ts":67.256, "dur":7.771, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"DMA", "ph":"X", "ts":235.947, "dur":9.148, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"DMA", "ph":"X", "ts":245.240, "dur":9.677, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"DMA", "ph":"X", "ts":255.067, "dur":10.525, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"DMA", "ph":"X", "ts":265.743, "dur":6.834, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]", "cat":"DMA", "ph":"X", "ts":272.721, "dur":4.697, "pid":1, "tid":2},
// CHECK: {"name":"pool1?t_Convolution/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]", "cat":"DMA", "ph":"X", "ts":277.586, "dur":0.678, "pid":1, "tid":2},
// CHECK: {"name":"pool1?t_Convolution", "cat":"DMA", "ph":"X", "ts":335.583, "dur":2.251, "pid":1, "tid":2},
// CHECK: {"name":"pool1?t_Convolution", "cat":"DMA", "ph":"X", "ts":338.093, "dur":2.081, "pid":1, "tid":2},
// CHECK: {"name":"pool1?t_Convolution", "cat":"DMA", "ph":"X", "ts":340.467, "dur":2.201, "pid":1, "tid":2},
// CHECK: {"name":"pool1?t_Convolution", "cat":"DMA", "ph":"X", "ts":342.833, "dur":1.454, "pid":1, "tid":2},
// CHECK: {"name":"output", "cat":"DMA", "ph":"X", "ts":344.543, "dur":20.367, "pid":1, "tid":2},
// CHECK: {"name":"output?_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":365.060, "dur":36.390, "pid":1, "tid":2},
// CHECK: {"name":"output?_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":401.609, "dur":36.395, "pid":1, "tid":2},
// CHECK: {"name":"output?_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":438.149, "dur":36.390, "pid":1, "tid":2},
// CHECK: {"name":"output?_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":474.681, "dur":18.977, "pid":1, "tid":2},
// CHECK: {"name":"output", "cat":"DMA", "ph":"X", "ts":493.844, "dur":7.147, "pid":1, "tid":2},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"SW", "ph":"X", "ts":24.211, "dur":37.698, "pid":1, "tid":4, "args":{"Stall cycles": "68247"}},
// CHECK: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"SW", "ph":"X", "ts":75.028, "dur":155.574, "pid":1, "tid":4, "args":{"Stall cycles": "1155920"}},
// CHECK: {"name":"output", "cat":"SW", "ph":"X", "ts":502.986, "dur":65.081, "pid":1, "tid":4, "args":{"Stall cycles": "233572"}},
// CHECK: {"name":"conv1/WithoutBiases", "cat":"Layer", "ph":"X", "ts":0.000, "dur":325.330, "pid":1, "tid":5, "args":{"Layer type": "Convolution"}},
// CHECK: {"name":"pool1", "cat":"Layer", "ph":"X", "ts":277.586, "dur":66.701, "pid":1, "tid":5, "args":{"Layer type": "Convolution"}},
// CHECK: {"name":"output", "cat":"Layer", "ph":"X", "ts":344.543, "dur":223.524, "pid":1, "tid":5, "args":{"Layer type": ""}},
// CHECK: {"name": "process_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name" : "Inference"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 2, "args": {"name" : "DMA"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 3, "args": {"name" : "DPU"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 4, "args": {"name" : "SW"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 5, "args": {"name" : "Layers"}}
// CHECK: ],
// CHECK: "displayTimeUnit": "ns"
// CHECK: }

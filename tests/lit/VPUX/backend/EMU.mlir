//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | vpux-translate --export-EMU -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json

module @Test attributes {VPU.compilationMode = "ReferenceSW"} {

module @UsedMemory {
    IE.MemoryResource 2048 bytes of @DDR
    IE.MemoryResource 1048576 bytes of @CMX_NN
}

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf32>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf32>
    }

func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %0: tensor<1x1x1x1000xf16>
}

}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           1,
// CHECK:           1,
// CHECK:           1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:           2.0,
// CHECK:           2000.0,
// CHECK:           2000.0,
// CHECK:           2000.0,
// CHECK:           2.0
// CHECK:       ],
// CHECK:       locale: "ProgrammableInput",
// CHECK:       locale_index: [
// CHECK:         0
// CHECK:       ],
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "softmax",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1,
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         2.0,
// CHECK:         2000.0,
// CHECK:         2000.0,
// CHECK:         2000.0,
// CHECK:         2.0
// CHECK:       ],
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 1,

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         4.0,
// CHECK:         4000.0,
// CHECK:         4.0
// CHECK:       ],
// CHECK:       locale: "ProgrammableInput",
// CHECK:       locale_index: [
// CHECK:         0
// CHECK:       ],
// CHECK:       data_dtype: "FP32"
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "softmax",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       strides: [
// CHECK:         4.0,
// CHECK:         4000.0,
// CHECK:         4.0
// CHECK:       ],
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       locale_index: [
// CHECK:         0
// CHECK:       ],
// CHECK:       data_dtype: "FP32"
// CHECK:     }
// CHECK:   ]
// CHECK:   task_lists: [
// CHECK:       {
// CHECK:         content: [
// CHECK:           {
// CHECK:             task_type: "UPALayerTask",
// CHECK:             task: {
// CHECK:               softLayerParams_type: "SoftmaxParams",
// CHECK:               softLayerParams: {
// CHECK:                 axis: 3
// CHECK:               },
// CHECK:               inputs: [
// CHECK:                 {
// CHECK:                   name: "input",
// CHECK:                   dimensions: [
// CHECK:                     1,
// CHECK:                     1,
// CHECK:                     1,
// CHECK:                     1000
// CHECK:                   ],
// CHECK:                   strides: [
// CHECK:                     2.0,
// CHECK:                     2000.0,
// CHECK:                     2000.0,
// CHECK:                     2000.0,
// CHECK:                     2.0
// CHECK:                   ],
// CHECK:                   locale: "ProgrammableInput",
// CHECK:                   locale_index: [
// CHECK:                     0
// CHECK:                   ],
// CHECK:                   data_dtype: "FP16",
// CHECK:                 }
// CHECK:               ],
// CHECK:               outputs: [
// CHECK:                 {
// CHECK:                   dimensions: [
// CHECK:                     1,
// CHECK:                     1,
// CHECK:                     1,
// CHECK:                     1000
// CHECK:                   ],
// CHECK:                   strides: [
// CHECK:                     2.0,
// CHECK:                     2000.0,
// CHECK:                     2000.0,
// CHECK:                     2000.0,
// CHECK:                     2.0
// CHECK:                   ],
// CHECK:                   locale: "ProgrammableOutput",
// CHECK:                   locale_index: [
// CHECK:                     0
// CHECK:                   ],
// CHECK:                   data_dtype: "FP16",



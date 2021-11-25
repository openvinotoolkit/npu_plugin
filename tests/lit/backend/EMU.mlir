// RUN: vpux-translate --export-EMU -o %t %s && flatc --raw-binary --json %vpuip_schema_file% -- %t && FileCheck %s --input-file %basename_t.json

module @Test attributes {VPUIP.arch = "KMB"} {

IERT.RunTimeResources
    availableMemory : {
        MemoryResource 1073741824 bytes
        MemoryResource 31457280 bytes of "DDR" {VPUIP.bandwidth = 8, VPUIP.derateFactor = 6.000000e-01}
        MemoryResource 4194304 bytes of "CMX_UPA" {VPUIP.bandwidth = 16, VPUIP.derateFactor = 8.500000e-01}
        MemoryResource 1048576 bytes of "CMX_NN" {VPUIP.bandwidth = 32, VPUIP.derateFactor = 1.000000e+00}
    }
    usedMemory : {
        MemoryResource 2048 bytes of "DDR"
        MemoryResource 1048576 bytes of "CMX_NN"
    }
    executors : {
        ExecutorResource 1 of "Leon_RT"
        ExecutorResource 1 of "Leon_NN"
        ExecutorResource 16 of "SHAVE_UPA"
        ExecutorResource 20 of "SHAVE_NN"
        ExecutorResource 4 of "NCE_Cluster" {
            ExecutorResource 5 of "NCE_PerClusterDPU"
        }
        ExecutorResource 1 of "DMA_UPA"
        ExecutorResource 1 of "DMA_NN"
    }

VPUIP.Graph
    options : "NONE"
    version : {
        majorV = 3,
        minorV = 11,
        patchV = 0,
        hash = "",
        contextStr = "VPUX Compiler"
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
    %0 = EMU.SoftMaxUPA {axisInd = 3} inputs(%arg0 : tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16>
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



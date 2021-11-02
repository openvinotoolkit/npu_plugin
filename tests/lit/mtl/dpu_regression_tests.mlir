// RUN: vpux-translate --split-input-file --import-HWTEST %s | FileCheck %s

// CHECK-LABEL: module @mainModule attributes {VPUIP.arch = "MTL", VPUIP.compilationMode = "ReferenceSW"}

{
    "case_type": "EltwiseAdd",
    "input": {
        "shape": [
            1,
            256,
            16,
            16
        ],
        "dtype": "uint8",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": 0,
            "high_range": 63
        }
    },
    "weight": {
        "shape": [
            1,
            256,
            16,
            16
        ],
        "dtype": "uint8",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": 0,
            "high_range": 63
        }
    },
    "output": {
        "shape": [
            1,
            256,
            16,
            16
        ],
        "dtype": "uint8",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": 0,
            "high_range": 255
        }
    },
    "activation": {
        "name": null
    }
}

// CHECK:       VPUIP.Graph
// CHECK-SAME:      options : "NONE"
// CHECK-SAME:      version : {
// CHECK-SAME:          contextStr = "VPUX Compiler"
// CHECK-SAME:      }

// CHECK:       IERT.RunTimeResources
// CHECK-SAME:      availableMemory :  {
// CHECK:               MemoryResource 524288000 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
// CHECK:               MemoryResource 1966080 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
// CHECK:           }
// CHECK-SAME:      usedMemory :  {
// CHECK:           }
// CHECK-SAME:      executors :  {
// CHECK:               ExecutorResource 2 of "DMA_NN"
// CHECK:               ExecutorResource {
// CHECK:                   ExecutorResource 1 of "NCE_PerClusterDPU"
// CHECK:               }
// CHECK:           }

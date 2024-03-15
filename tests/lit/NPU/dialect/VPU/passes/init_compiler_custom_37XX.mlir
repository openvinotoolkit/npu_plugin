//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-VPUX37XX

// CHECK: module @mode attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>}
module @mode attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>} {
}

// -----

// CHECK: module @arch attributes {VPU.arch = #VPU.arch_kind<VPUX30XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>}
module @arch attributes {VPU.arch = #VPU.arch_kind<VPUX30XX>} {
}

// -----

// CHECK: module @executors attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>}
module @executors {
    IE.ExecutorResource 5 of @DMA_NN
    IE.TileResource 5 of @NCE at 6.000000e+02 MHz
}

// CHECK-DAG:    {{  }}IE.ExecutorResource 5 of @DMA_NN
// CHECK-DAG:    {{  }}IE.TileResource 5 of @NCE at 6.000000e+02 MHz {
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @DPU
// CHECK-DAG:    {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @SHAVE_NN
// CHECK-DAG:    {{    }}IE.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
// CHECK-DAG:    {{    }}IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
// CHECK-DAG:    {{  }}IE.MemoryResource 2306867200 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}


// -----

// CHECK: module @memory attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>}
module @memory {
    IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
        IE.MemoryResource 5 bytes of @CMX_NN_FragmentationAware
        IE.MemoryResource 10000 bytes of @CMX_NN {VPU.bandwidth = 10 : i64, VPU.derateFactor = 2.0 : f64}
    }
    IE.MemoryResource 500000 bytes of @DDR
}

// CHECK-DAG:    {{  }}IE.ExecutorResource 2 of @DMA_NN
// CHECK-DAG:    {{  }}IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
// CHECK-DAG:    {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @SHAVE_NN
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @DPU
// CHECK-DAG:    {{    }}IE.MemoryResource 5 bytes of @CMX_NN_FragmentationAware
// CHECK-DAG:    {{    }}IE.MemoryResource 10000 bytes of @CMX_NN {VPU.bandwidth = 10 : i64, VPU.derateFactor = 2.000000e+00 : f64}
// CHECK-DAG:    {{  }}IE.MemoryResource 500000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}

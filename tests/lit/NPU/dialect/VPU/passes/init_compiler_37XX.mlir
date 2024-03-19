//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceSW" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-VPUX37XX

// CHECK: module @test attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>}
module @test {

// CHECK-DAG:    {{  }}IE.ExecutorResource 2 of @DMA_NN
// CHECK-DAG:    {{  }}IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
// CHECK-DAG:    {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @SHAVE_NN
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @DPU
// CHECK-DAG:    {{    }}IE.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
// CHECK-DAG:    {{    }}IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
// CHECK-DAG:    {{  }}IE.MemoryResource 2306867200 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}

}

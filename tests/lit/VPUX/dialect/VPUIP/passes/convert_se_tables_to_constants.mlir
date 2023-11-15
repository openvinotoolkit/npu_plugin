//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-se-tables-to-constants %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SETableInterpolateNearest() -> memref<1x1x6x6xi32, #NHWC> {
    %0 = VPUIP.StorageElementTable {
            basePtrs = dense<[0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1]> : tensor<36xi32>,
            dataElemType = f16,
            dataShape = [1, 32, 3, 3],
            seAttr = #VPU.SEInterpolate<mode = <NEAREST>,
                                        coordinate_transformation_mode = <ASYMMETRIC>,
                                        scale = [1.0, 1.0, 2.0, 2.0],
                                        nearest_mode = <FLOOR>,
                                        offsets = [0, 0, 0, 0],
                                        sizes = [1, 32, 6, 6]>,
            seDepth = 1 : i64,
            seSize = 32 : i64
        } -> memref<1x1x6x6xi32, #NHWC>
    return %0 : memref<1x1x6x6xi32, #NHWC>

    // Pointers have the following offsets:
    //     0   0  64  64 128 128
    //     0   0  64  64 128 128
    //   192 192 256 256 320 320
    //   192 192 256 256 320 320
    //   384 384 448 448 512 512
    //   384 384 448 448 512 512
    // The offsets are reset for separate clusters:
    //     0   0  64  64 128 128
    //     0   0  64  64 128 128
    //   192 192 256 256 320 320
    //   192 192 256 256 320 320
    //     0   0  64  64 128 128
    //     0   0  64  64 128 128
    // Without the last 4 bits:
    //      0  0  4  4  8  8
    //      0  0  4  4  8  8
    //     12 12 16 16 20 20
    //     12 12 16 16 20 20
    //      0  0  4  4  8  8
    //      0  0  4  4  8  8
    // Shifted left 9 times:
    //        0     0  2048  2048  4096  4096
    //        0     0  2048  2048  4096  4096
    //     6144  6144  8192  8192 10240 10240
    //     6144  6144  8192  8192 10240 10240
    //        0     0  2048  2048  4096  4096
    //        0     0  2048  2048  4096  4096
    // Then, the base_ptrs values are added to the last 9 bits.

    // CHECK-NOT:            VPUIP.StorageElementTable
    // CHECK:                [[CST:%.+]] = const.Declare memref<1x1x6x6xi32, #NHWC>
    // CHECK-SAME{LITERAL}:    = dense<[[[[0, 0, 2048, 2048, 4096, 4096],
    // CHECK-SAME{LITERAL}:               [0, 0, 2048, 2048, 4096, 4096],
    // CHECK-SAME{LITERAL}:               [6144, 6144, 8192, 8192, 10240, 10240],
    // CHECK-SAME{LITERAL}:               [6144, 6144, 8192, 8192, 10240, 10240],
    // CHECK-SAME{LITERAL}:               [1, 1, 2049, 2049, 4097, 4097],
    // CHECK-SAME{LITERAL}:               [1, 1, 2049, 2049, 4097, 4097]]]]> : tensor<1x1x6x6xi32, {order = #NHWC}>
    // CHECK:                return [[CST]] : memref<1x1x6x6xi32, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SETableInterpolateBilinear() -> memref<1x1x7x7xi32, #NHWC> {
    %0 = VPUIP.StorageElementTable {
            basePtrs = dense<[0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0,
                              1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1]> : tensor<49xi32>,
            dataElemType = f16,
            dataShape = [1, 32, 3, 3],
            seAttr = #VPU.SEInterpolate<mode = <BILINEAR>,
                                        coordinate_transformation_mode = <ASYMMETRIC>,
                                        scale = [1.0, 1.0, 2.0, 2.0],
                                        offsets = [0, 0, 0, 0],
                                        sizes = [1, 32, 7, 7]>,
            seDepth = 1 : i64,
            seSize = 32 : i64
        } -> memref<1x1x7x7xi32, #NHWC>
    return %0 : memref<1x1x7x7xi32, #NHWC>

    // Pointers have the following offsets:
    //     0   0  64  64 128 128 128
    //     0   0  64  64 128 128 128
    //   192 192 256 256 320 320 320
    //   192 192 256 256 320 320 320
    //   384 384 448 448 512 512 512
    //   384 384 448 448 512 512 512
    //   384 384 448 448 512 512 512
    // The offsets are reset for separate clusters:
    //     0   0  64  64 128 128 128
    //     0   0  64  64 128 128 128
    //   192 192 256 256 320 320 320
    //     0   0  64  64 128 128 128
    //   192 192 256 256 320 320 320
    //   192 192 256 256 320 320 320
    //   192 192 256 256 320 320 320
    // Without the last 4 bits:
    //      0  0  4  4  8  8  8
    //      0  0  4  4  8  8  8
    //     12 12 16 16 20 20 20
    //      0  0  4  4  8  8  8
    //     12 12 16 16 20 20 20
    //     12 12 16 16 20 20 20
    //     12 12 16 16 20 20 20
    // Shifted left 9 times:
    //        0     0  2048  2048  4096  4096  4096
    //        0     0  2048  2048  4096  4096  4096
    //     6144  6144  8192  8192 10240 10240 10240
    //        0     0  2048  2048  4096  4096  4096
    //     6144  6144  8192  8192 10240 10240 10240
    //     6144  6144  8192  8192 10240 10240 10240
    //     6144  6144  8192  8192 10240 10240 10240
    // Then, the base_ptrs values are added to the last 9 bits.

    // CHECK-NOT:            VPUIP.StorageElementTable
    // CHECK:                [[CST:%.+]] = const.Declare memref<1x1x7x7xi32, #NHWC>
    // CHECK-SAME{LITERAL}:    = dense<[[[[0, 0, 2048, 2048, 4096, 4096, 4096],
    // CHECK-SAME{LITERAL}:               [0, 0, 2048, 2048, 4096, 4096, 4096],
    // CHECK-SAME{LITERAL}:               [6144, 6144, 8192, 8192, 10240, 10240, 10240],
    // CHECK-SAME{LITERAL}:               [1, 1, 2049, 2049, 4097, 4097, 4097],
    // CHECK-SAME{LITERAL}:               [6145, 6145, 8193, 8193, 10241, 10241, 10241],
    // CHECK-SAME{LITERAL}:               [6145, 6145, 8193, 8193, 10241, 10241, 10241],
    // CHECK-SAME{LITERAL}:               [6145, 6145, 8193, 8193, 10241, 10241, 10241]]]]> : tensor<1x1x7x7xi32, {order = #NHWC}>
    // CHECK:                return [[CST]] : memref<1x1x7x7xi32, #NHWC>
}

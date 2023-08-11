//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

func.func @ParsePrintOpaqueConst() -> tensor<1x11x1x1xf16> {
    %cst = const.Declare tensor<1x11x1x1xf16> =
                #const.OpaqueElements<"0x000000000000803E0000003F0000403F0000803F0000A03F0000C03F0000E03F000000400000104000002040"> :
                    tensor<1x11x1x1xf32>, [#const.ConvertElemType<f16>]

    return %cst : tensor<1x11x1x1xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<1x11x1x1xf16>
    // CHECK-SAME:       #const.OpaqueElements<"0x000000000000803E0000003F0000403F0000803F0000A03F0000C03F0000E03F000000400000104000002040">
    // CHECK-SAME:       tensor<1x11x1x1xf32>, [#const.ConvertElemType<f16>]

    // CHECK:       return [[CST]]
}

// -----

func.func @ParsePrintDenseConst() -> tensor<2xf16> {
    %cst = const.Declare tensor<2xf16> = dense<[1.0, 2.0]> : tensor<2xf16>

    return %cst : tensor<2xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<2xf16> = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf16>
    // CHECK:       return [[CST]]
}

// -----

func.func @CanNotParseElidedConst() -> tensor<11x3x2x2xf16> {
    // expected-error@+2 {{custom op 'const.Declare' Size of opaque buffer '4' in 'OpaqueElementsAttr' doesn't match its Type 'tensor<11x3x2x2xf32>'}}
    // expected-error@+1 {{custom op 'const.Declare' Failed to parse content attribute}}
    %cst_0 = const.Declare tensor<11x3x2x2xf16> = #const.OpaqueElements<"elided_large_const", "0xDEADBEEF"> :
                tensor<11x3x2x2xf32>, [#const.ConvertElemType<f16>]

    return %cst_0 : tensor<11x3x2x2xf16>
}

// -----

func.func @CanNotParseWronngStartOfHex() -> tensor<11x3x2x2xf16> {
    // expected-error@+2 {{Hex string should start with `0x`}}
    // expected-error@+1 {{custom op 'const.Declare' Failed to parse content attribute}}
    %cst_0 = const.Declare tensor<11x3x2x2xf16> = #const.OpaqueElements<"000000000000803E0000003F0000403F0000803F0000A03F0000C03F0000E03F000000400000104000002040"> :
                tensor<11x3x2x2xf32>, [#const.ConvertElemType<f16>]

    return %cst_0 : tensor<11x3x2x2xf16>
}

// -----

func.func @CanNotParseEmptyValue() -> tensor<11x3x2x2xf16> {
    // expected-error@+2 {{custom op 'const.Declare' Size of opaque buffer '0' in 'OpaqueElementsAttr' doesn't match its Type 'tensor<11x3x2x2xf32>'}}
    // expected-error@+1 {{custom op 'const.Declare' Failed to parse content attribute}}
    %cst_0 = const.Declare tensor<11x3x2x2xf16> = #const.OpaqueElements<"0x"> :
                tensor<11x3x2x2xf32>, [#const.ConvertElemType<f16>]

    return %cst_0 : tensor<11x3x2x2xf16>
}

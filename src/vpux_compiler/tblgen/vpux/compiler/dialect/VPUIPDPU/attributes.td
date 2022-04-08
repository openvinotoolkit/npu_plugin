//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPUIPDPU_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/VPUIPDPU/dialect.td"

//
// PPEBypassMode
//

def VPUIPDPU_PPEBypassMode :
        I32EnumAttr<
            "PPEBypassMode",
            "PPE bypass mode",
            [
                I32EnumAttrCase<"BypassOff", 0>,
                I32EnumAttrCase<"BypassOn", 1>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// PPEFpConvertMode
//

def VPUIPDPU_PPEFpConvertMode :
        I32EnumAttr<
            "PPEFpConvertMode",
            "PPE FP convert mode",
            [
                I32EnumAttrCase<"FpConv_Bypass", 0>,
                I32EnumAttrCase<"FpConv_Fp16_RNE", 1>,
                I32EnumAttrCase<"FpConv_Bfp16", 2>,
                I32EnumAttrCase<"FpConv_Fp8_RNE", 3>,
                I32EnumAttrCase<"FpConv_I32_RNE", 4>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// PPEFpConvClampMode
//

def VPUIPDPU_PPEFpConvClampMode :
        I32EnumAttr<
            "PPEFpConvClampMode",
            "PPE FP convert clamp mode",
            [
                I32EnumAttrCase<"FpConv_Clamp_Off", 0>,
                I32EnumAttrCase<"FpConv_Clamp_On", 1>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// PPEFpConvFTZMode
//

def VPUIPDPU_PPEFpConvFTZMode :
        I32EnumAttr<
            "PPEFpConvFTZMode",
            "PPE FP convert force-to-zero mode",
            [
                I32EnumAttrCase<"FpConv_FTZ_Off", 0>,
                I32EnumAttrCase<"FpConv_FTZ_On", 1>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// PPEFpConvBf16RoundMode
//

def VPUIPDPU_PPEFpConvBf16RoundMode :
        I32EnumAttr<
            "PPEFpConvBf16RoundMode",
            "PPE FP convert to BF16 round mode",
            [
                I32EnumAttrCase<"FpConv_BF16_Truncate", 0>,
                I32EnumAttrCase<"FpConv_BF16_RNE", 1>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// PPEIntRoundMode
//

def VPUIPDPU_PPEIntRoundMode :
        I32EnumAttr<
            "PPEIntRoundMode",
            "PPE Int round mode",
            [
                I32EnumAttrCase<"IntRound_TiesToEven", 0>,
                I32EnumAttrCase<"IntRound_TiesAwayFromZero", 1>,
                I32EnumAttrCase<"IntRound_Bypass", 2>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// PPEIntConvertMode
//

def VPUIPDPU_PPEIntConvertMode :
        I32EnumAttr<
            "PPEIntConvertMode",
            "PPE Int convert mode",
            [
                I32EnumAttrCase<"IntConv_Bypass", 0>,
                I32EnumAttrCase<"IntConv_Fp16_RNE", 1>,
                I32EnumAttrCase<"IntConv_Fp8_RNE", 2>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUDataTypes
//

def VPUIPDPU_ODUDataTypes :
        I32EnumAttr<
            "ODUDataTypes",
            "ODU output data types - VPU2.7",
            [
                I32EnumAttrCase<"ODU_DTYPE_FP16", 0>,
                I32EnumAttrCase<"ODU_DTYPE_U8F",  1>,
                I32EnumAttrCase<"ODU_DTYPE_G8",   2>,
                I32EnumAttrCase<"ODU_DTYPE_I8",   3>,
                I32EnumAttrCase<"ODU_DTYPE_I32",  4>,
                I32EnumAttrCase<"ODU_DTYPE_I4",   5>,
                I32EnumAttrCase<"ODU_DTYPE_I2",   6>,
                I32EnumAttrCase<"ODU_DTYPE_LOG",  7>,
                I32EnumAttrCase<"ODU_DTYPE_BIN",  8>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUOutTensorDataMode
//

def VPUIPDPU_ODUActivationReuseMode :
        I32EnumAttr<
            "ODUActivationReuseMode",
            "ODU activation reuse mode",
            [
                I32EnumAttrCase<"NTHW_1",  0>,
                I32EnumAttrCase<"NTHW_4",  1>,
                I32EnumAttrCase<"NTHW_8",  2>,
                I32EnumAttrCase<"NTHW_16", 3>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUPermuteDataMode
//

def VPUIPDPU_ODUPermuteDataMode :
        I32EnumAttr<
            "ODUPermuteDataMode",
            "ODU data permutation mode",
            [
                I32EnumAttrCase<"PERMUTE_ZXY",  0>,
                I32EnumAttrCase<"PERMUTE_ZYX",  1>,
                I32EnumAttrCase<"PERMUTE_YZX",  2>,
                I32EnumAttrCase<"PERMUTE_YXZ",  3>,
                I32EnumAttrCase<"PERMUTE_XZY",  4>,
                I32EnumAttrCase<"PERMUTE_XYZ",  5>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUMemoryMode
//

def VPUIPDPU_ODUMemoryMode :
        I32EnumAttr<
            "ODUMemoryMode",
            "ODU memory data mode",
            [
                I32EnumAttrCase<"MODE_DENSE",  0>,
                I32EnumAttrCase<"MODE_SUPERDENSE",  1>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUWcbCombineMode
//

def VPUIPDPU_ODUWcbCombineMode :
        I32EnumAttr<
            "ODUWcbCombineMode",
            "ODU WCB combination mode",
            [
                I32EnumAttrCase<"WCB_COMBINE_BY_CONTEXT", 0>,
                I32EnumAttrCase<"WCB_COMBINE_BY_ADDRESS", 1>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUSwizzleMode
//

def VPUIPDPU_ODUSwizzleKey :
        I32EnumAttr<
            "ODUSwizzleKey",
            "ODU out data swizzle key config.",
            [
                I32EnumAttrCase<"SWIZZLE_OFF",   0>,
                I32EnumAttrCase<"SWIZZLE_KEY_1", 1>,
                I32EnumAttrCase<"SWIZZLE_KEY_2", 2>,
                I32EnumAttrCase<"SWIZZLE_KEY_3", 3>,
                I32EnumAttrCase<"SWIZZLE_KEY_4", 4>,
                I32EnumAttrCase<"SWIZZLE_KEY_5", 5>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

//
// ODUHaloAugmentatorDPUTile
//

def VPUIPDPU_DPUTiles :
        BitEnumAttr<
            "DPUTiles",
            "DPU tile IDs",
            [
                BitEnumAttrCase<"DPU_TILE_0", 1>,
                BitEnumAttrCase<"DPU_TILE_1", 2>,
                BitEnumAttrCase<"DPU_TILE_2", 4>,
                BitEnumAttrCase<"DPU_TILE_3", 8>,
                BitEnumAttrCase<"DPU_TILE_4", 16>,
                BitEnumAttrCase<"DPU_TILE_5", 32>
            ]
        > {
    let cppNamespace = "vpux::VPUIPDPU";
    let genSpecializedAttr = 1;
}

#endif

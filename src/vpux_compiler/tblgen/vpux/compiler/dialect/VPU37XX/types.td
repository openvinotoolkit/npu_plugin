//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#ifndef VPUX_COMPILER_DIALECT_VPU37XX_TYPES
#define VPUX_COMPILER_DIALECT_VPU37XX_TYPES

include "vpux/compiler/dialect/VPURegMapped/dialect.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"
include "vpux/compiler/dialect/VPU37XX/dialect.td"

//
// RegField Specialization
//

class VPU37XX_RegFieldWrapper<string name = "", int width = 0, int pos = 0> :
    VPURegMapped_RegFieldWrapper<VPU37XX_Dialect, name, width, pos> {

    }

//
// DPUInvariantRegisters
//

// z_config_bf
def bf_SeZSplit : VPU37XX_RegFieldWrapper<"SeZSplit", 4, 0>;
def bf_NumSesInZDir : VPU37XX_RegFieldWrapper<"NumSesInZDir", 9, 4>;
def bf_CmSpPattern : VPU37XX_RegFieldWrapper<"CmSpPattern", 16, 13>;
def bf_Reserved : VPU37XX_RegFieldWrapper<"Reserved", 2, 29>;
def bf_AddrFormatSel : VPU37XX_RegFieldWrapper<"AddrFormatSel", 1, 31>;

// kernel_pad_cfg_bf
def bf_MpeAssign : VPU37XX_RegFieldWrapper<"MpeAssign", 1, 0>;
def bf_PadRightEn : VPU37XX_RegFieldWrapper<"PadRightEn", 1, 1>;
def bf_PadLeftEn : VPU37XX_RegFieldWrapper<"PadLeftEn", 1, 2>;
def bf_PadBottomEn : VPU37XX_RegFieldWrapper<"PadBottomEn", 1, 3>;
def bf_PadTopEn : VPU37XX_RegFieldWrapper<"PadTopEn", 1, 4>;
def bf_KernelY : VPU37XX_RegFieldWrapper<"KernelY", 4, 5>;
def bf_KernelX : VPU37XX_RegFieldWrapper<"KernelX", 4, 9>;
def bf_WtPltCfg : VPU37XX_RegFieldWrapper<"WtPltCfg", 2, 13>;
def bf_ActDense : VPU37XX_RegFieldWrapper<"ActDense", 1, 15>;
def bf_WtDense : VPU37XX_RegFieldWrapper<"WtDense", 1, 16>;
def bf_StrideYEn : VPU37XX_RegFieldWrapper<"StrideYEn", 1, 17>;
def bf_StrideY : VPU37XX_RegFieldWrapper<"StrideY", 3, 18>;
def bf_DynamicBwEn : VPU37XX_RegFieldWrapper<"DynamicBwEn", 1, 21>;
def bf_DwWtSpIns : VPU37XX_RegFieldWrapper<"DwWtSpIns", 1, 22>;
def bf_Layer1WtSpIns : VPU37XX_RegFieldWrapper<"Layer1WtSpIns", 1, 23>;
def bf_Layer1CmpEn : VPU37XX_RegFieldWrapper<"Layer1CmpEn", 1, 24>;
def bf_PoolOptEn : VPU37XX_RegFieldWrapper<"PoolOptEn", 1, 25>;
def bf_Unused1 : VPU37XX_RegFieldWrapper<"Unused1", 3, 26>;
def bf_SpSeTblSegment : VPU37XX_RegFieldWrapper<"SpSeTblSegment", 1, 29>;
def bf_RstCtxt : VPU37XX_RegFieldWrapper<"RstCtxt", 1, 30>;
def bf_Unused2 : VPU37XX_RegFieldWrapper<"Unused2", 1, 31>;

// tensor_size0_bf
def bf_TensorSizeX : VPU37XX_RegFieldWrapper<"TensorSizeX", 14, 0>;
def bf_TensorSizeY : VPU37XX_RegFieldWrapper<"TensorSizeY", 14, 14>;

// tensor_size1_bf
def bf_TensorSizeZ : VPU37XX_RegFieldWrapper<"TensorSizeZ", 14, 0>;

// tensor_mode_bf
def bf_Wmode : VPU37XX_RegFieldWrapper<"Wmode", 4, 0>;
def bf_Amode : VPU37XX_RegFieldWrapper<"Amode", 4, 4>;
def bf_Stride : VPU37XX_RegFieldWrapper<"Stride", 3, 8>;
def bf_ZmInput : VPU37XX_RegFieldWrapper<"ZmInput", 1, 11>;
def bf_DwInput : VPU37XX_RegFieldWrapper<"DwInput", 1, 12>;
def bf_CmInput : VPU37XX_RegFieldWrapper<"CmInput", 1, 13>;
def bf_WorkloadOperation : VPU37XX_RegFieldWrapper<"WorkloadOperation", 2, 14>;
def bf_PadValue : VPU37XX_RegFieldWrapper<"PadValue", 16, 16>;

// elops_wload_bf
def bf_ElopWload : VPU37XX_RegFieldWrapper<"ElopWload", 1, 0>;
def bf_SeedWload : VPU37XX_RegFieldWrapper<"SeedWload", 1, 1>;
def bf_FifoWrWload : VPU37XX_RegFieldWrapper<"FifoWrWload", 1, 2>;
def bf_ElopWloadType : VPU37XX_RegFieldWrapper<"ElopWloadType", 1, 3>;
def bf_PoolWtData : VPU37XX_RegFieldWrapper<"PoolWtData", 16, 4>;
def bf_PoolWtRdDis : VPU37XX_RegFieldWrapper<"PoolWtRdDis", 1, 20>;


// odu_cfg_bf
def bf_Dtype : VPU37XX_RegFieldWrapper<"Dtype", 4, 0>;
def bf_SpValue : VPU37XX_RegFieldWrapper<"SpValue", 8, 5>;
def bf_SpOutEn : VPU37XX_RegFieldWrapper<"SpOutEn", 1, 13>;
def bf_WriteSp : VPU37XX_RegFieldWrapper<"WriteSp", 1, 15>;
def bf_WritePt : VPU37XX_RegFieldWrapper<"WritePt", 1, 16>;
def bf_WriteAc : VPU37XX_RegFieldWrapper<"WriteAc", 1, 17>;
def bf_Mode : VPU37XX_RegFieldWrapper<"Mode", 2, 18>;
def bf_Grid : VPU37XX_RegFieldWrapper<"Grid", 1, 20>;
def bf_SwizzleKey : VPU37XX_RegFieldWrapper<"SwizzleKey", 3, 21>;
def bf_Nthw : VPU37XX_RegFieldWrapper<"Nthw", 2, 25>;
def bf_Permutation : VPU37XX_RegFieldWrapper<"Permutation", 3, 27>;
def bf_DebugMode : VPU37XX_RegFieldWrapper<"DebugMode", 1, 30>;

// te_dim0_bf
def bf_TeDimY : VPU37XX_RegFieldWrapper<"TeDimY", 13, 0>;
def bf_TeDimZ : VPU37XX_RegFieldWrapper<"TeDimZ", 13, 13>;

// odu_cast_bf
def bf_CastEnable : VPU37XX_RegFieldWrapper<"CastEnable", 1, 0>;
def bf_CastOffset : VPU37XX_RegFieldWrapper<"CastOffset", 28, 4>;

// mpe_cfg_bf
def bf_MpeWtbias : VPU37XX_RegFieldWrapper<"MpeWtbias", 8, 0>;
def bf_MpeActbias : VPU37XX_RegFieldWrapper<"MpeActbias", 8, 8>;
def bf_MpeMode : VPU37XX_RegFieldWrapper<"MpeMode", 3, 16>;
def bf_MpeDense : VPU37XX_RegFieldWrapper<"MpeDense", 1, 19>;
def bf_MrmWeightDense : VPU37XX_RegFieldWrapper<"MrmWeightDense", 1, 20>;
def bf_MrmActDense : VPU37XX_RegFieldWrapper<"MrmActDense", 1, 21>;
def bf_MpeDaz : VPU37XX_RegFieldWrapper<"MpeDaz", 1, 22>;
def bf_MpeFtz : VPU37XX_RegFieldWrapper<"MpeFtz", 1, 23>;

// elop_scale_bf
def bf_ElopScaleB : VPU37XX_RegFieldWrapper<"ElopScaleB", 16, 0>;
def bf_ElopScaleA : VPU37XX_RegFieldWrapper<"ElopScaleA", 16, 16>;

// ppe_cfg_bf
def bf_PpeG8BiasC : VPU37XX_RegFieldWrapper<"PpeG8BiasC", 9, 0>;
def bf_PpeG8BiasB : VPU37XX_RegFieldWrapper<"PpeG8BiasB", 9, 9>;
def bf_PpeG8BiasA : VPU37XX_RegFieldWrapper<"PpeG8BiasA", 9, 18>;

// ppe_scale_bf
def bf_PpeScaleShift : VPU37XX_RegFieldWrapper<"PpeScaleShift", 6, 2>;
def bf_PpeScaleRound : VPU37XX_RegFieldWrapper<"PpeScaleRound", 2, 10>;
def bf_PpeScaleMult : VPU37XX_RegFieldWrapper<"PpeScaleMult", 16, 16>;

// ppe_scale_ctrl_bf
def bf_PpeScaleOverride : VPU37XX_RegFieldWrapper<"PpeScaleOverride", 1, 0>;
def bf_PpeFpScaleOverride : VPU37XX_RegFieldWrapper<"PpeFpScaleOverride", 1, 1>;

// ppe_prelu_bf
def bf_PpePreluShift : VPU37XX_RegFieldWrapper<"PpePreluShift", 5, 8>;
def bf_PpePreluMult : VPU37XX_RegFieldWrapper<"PpePreluMult", 11, 16>;

// ppe_misc_bf
def bf_PpeFp16Ftz : VPU37XX_RegFieldWrapper<"PpeFp16Ftz", 1, 6>;
def bf_PpeFp16Clamp : VPU37XX_RegFieldWrapper<"PpeFp16Clamp", 1, 7>;
def bf_PpeI32Convert : VPU37XX_RegFieldWrapper<"PpeI32Convert", 2, 8>;

// ppe_fp_cfg_bf
def bf_PpeFpConvert : VPU37XX_RegFieldWrapper<"PpeFpConvert", 3, 0>;
def bf_PpeFpBypass : VPU37XX_RegFieldWrapper<"PpeFpBypass", 1, 3>;
def bf_PpeBf16Round : VPU37XX_RegFieldWrapper<"PpeBf16Round", 1, 4>;
def bf_PpeFpPreluEn : VPU37XX_RegFieldWrapper<"PpeFpPreluEn", 1, 5>;

//
// DPUVariantRegisters
//

// workload_size0_bf
def bf_WorkloadSizeX : VPU37XX_RegFieldWrapper<"WorkloadSizeX", 14, 0>;
def bf_WorkloadSizeY : VPU37XX_RegFieldWrapper<"WorkloadSizeY", 14, 14>;

// workload_size1_bf
def bf_WorkloadSizeZ : VPU37XX_RegFieldWrapper<"WorkloadSizeZ", 14, 0>;
def bf_PadCountUp : VPU37XX_RegFieldWrapper<"PadCountUp", 3, 14>;
def bf_PadCountLeft : VPU37XX_RegFieldWrapper<"PadCountLeft", 3, 17>;
def bf_PadCountDown : VPU37XX_RegFieldWrapper<"PadCountDown", 3, 20>;
def bf_PadCountRight : VPU37XX_RegFieldWrapper<"PadCountRight", 3, 23>;

// workload_start0_bf
def bf_WorkloadStartX : VPU37XX_RegFieldWrapper<"WorkloadStartX", 14, 0>;
def bf_WorkloadStartY : VPU37XX_RegFieldWrapper<"WorkloadStartY", 14, 14>;

// workload_start1_bf
def bf_WorkloadStartZ : VPU37XX_RegFieldWrapper<"WorkloadStartZ", 14, 0>;

// offset_addr_bf
def bf_NthwNtk : VPU37XX_RegFieldWrapper<"NthwNtk", 2, 0>;
def bf_BinCfg : VPU37XX_RegFieldWrapper<"BinCfg", 1, 2>;
def bf_ConvCond : VPU37XX_RegFieldWrapper<"ConvCond", 1, 3>;
def bf_DenseSe : VPU37XX_RegFieldWrapper<"DenseSe", 1, 4>;
def bf_IdxQuad : VPU37XX_RegFieldWrapper<"IdxQuad", 1, 5>;
def bf_SwizzleKeyOffsetAddr : VPU37XX_RegFieldWrapper<"SwizzleKeyOffsetAddr", 3, 6>;
def bf_IduMrmClkEn : VPU37XX_RegFieldWrapper<"IduMrmClkEn", 1, 9>;
def bf_OduClkEn : VPU37XX_RegFieldWrapper<"OduClkEn", 1, 10>;
def bf_MpeClkEn : VPU37XX_RegFieldWrapper<"MpeClkEn", 1, 11>;
def bf_PpeClkEn : VPU37XX_RegFieldWrapper<"PpeClkEn", 1, 12>;
def bf_OduStatEn : VPU37XX_RegFieldWrapper<"OduStatEn", 1, 13>;
def bf_IduStatEn : VPU37XX_RegFieldWrapper<"IduStatEn", 1, 14>;
def bf_OduStatClrMode : VPU37XX_RegFieldWrapper<"OduStatClrMode", 1, 16>;
def bf_IduStatClrMode : VPU37XX_RegFieldWrapper<"IduStatClrMode", 1, 17>;
def bf_ShaveL2CacheEn : VPU37XX_RegFieldWrapper<"ShaveL2CacheEn", 1, 19>;
def bf_IduDbgEn : VPU37XX_RegFieldWrapper<"IduDbgEn", 2, 20>;
def bf_WtSwizzleKey : VPU37XX_RegFieldWrapper<"WtSwizzleKey", 3, 27>;
def bf_WtSwizzleSel : VPU37XX_RegFieldWrapper<"WtSwizzleSel", 1, 30>;

// te_end0_bf
def bf_TeEndY : VPU37XX_RegFieldWrapper<"TeEndY", 13, 0>;
def bf_TeEndZ : VPU37XX_RegFieldWrapper<"TeEndZ", 13, 13>;

// te_end1_bf
def bf_TeEndX : VPU37XX_RegFieldWrapper<"TeEndX", 13, 0>;

// te_beg0_bf
def bf_TeBegY : VPU37XX_RegFieldWrapper<"TeBegY", 13, 0>;
def bf_TeBegZ : VPU37XX_RegFieldWrapper<"TeBegZ", 13, 13>;

// te_beg1_bf
def bf_TeBegX : VPU37XX_RegFieldWrapper<"TeBegX", 13, 0>;

//
// Register Specialization
//

class VPU37XX_RegisterWrapper<string name = "", int size = 0, int address = 0x00, list<string> regFieldTypes = []> :
    VPURegMapped_RegisterWrapper<VPU37XX_Dialect, "vpux::VPU37XX::", name, size, address, regFieldTypes> {

}

//
// DPUInvariantRegisters
//

// z_Config
def reg_ZConfig : VPU37XX_RegisterWrapper<"ZConfig", 32, 0xac,
    ["SeZSplit", "NumSesInZDir", "CmSpPattern", "Reserved", "AddrFormatSel"]>;

// kernel_pad_cfg
def reg_KernelPadCfg : VPU37XX_RegisterWrapper<"KernelPadCfg", 32, 0xb0,
    ["MpeAssign", "PadRightEn", "PadLeftEn", "PadBottomEn", "PadTopEn", "KernelY", "KernelX", "WtPltCfg", "ActDense", "WtDense",
    "StrideYEn", "StrideY", "DynamicBwEn", "DwWtSpIns", "Layer1WtSpIns", "Layer1CmpEn", "PoolOptEn", "SpSeTblSegment", "RstCtxt"]>;

// tensor_size0
def reg_TensorSize0 : VPU37XX_RegisterWrapper<"TensorSize0", 32, 0xc0, ["TensorSizeX", "TensorSizeY"]>;

// tensor_size1
def reg_TensorSize1 : VPU37XX_RegisterWrapper<"TensorSize1", 32, 0xc4, ["TensorSizeZ"]>;

// tensor_mode
def reg_TensorMode : VPU37XX_RegisterWrapper<"TensorMode", 32, 0xcc,
    ["Wmode", "Amode", "Stride", "ZmInput", "DwInput", "CmInput", "WorkloadOperation", "PadValue"]>;

// elops_wload
def reg_ElopsWload : VPU37XX_RegisterWrapper<"ElopsWload", 32, 0xd8,
    ["ElopWload", "SeedWload", "FifoWrWload", "ElopWloadType", "PoolWtData", "PoolWtRdDis"]>;

// odu_cfg
def reg_OduCfg : VPU37XX_RegisterWrapper<"OduCfg", 32, 0xf8,
    ["Dtype", "SpValue", "SpOutEn", "WriteSp", "WritePt", "WriteAc", "Mode", "Grid", "SwizzleKey", "Nthw", "Permutation", "DebugMode"]>;

// te_dim0
def reg_TeDim0 : VPU37XX_RegisterWrapper<"TeDim0", 32, 0x108, ["TeDimY", "TeDimZ"]>;

// odu_cast
def reg_OduCast : VPU37XX_RegisterWrapper<"OduCast", 32, 0x130, ["CastEnable", "CastOffset"]>;

// mpe_cfg
def reg_MpeCfg : VPU37XX_RegisterWrapper<"MpeCfg", 32, 0x13c,
    ["MpeWtbias", "MpeActbias", "MpeMode", "MpeDense", "MrmWeightDense", "MrmActDense", "MpeDaz", "MpeFtz"]>;

// elop_scale
def reg_ElopScale : VPU37XX_RegisterWrapper<"ElopScale", 32, 0x144, ["ElopScaleB", "ElopScaleA"]>;

// ppe_cfg
def reg_PpeCfg : VPU37XX_RegisterWrapper<"PpeCfg", 32, 0x148, ["PpeG8BiasC", "PpeG8BiasB", "PpeG8BiasA"]>;

// ppe_scale
def reg_PpeScale : VPU37XX_RegisterWrapper<"PpeScale", 32, 0x150, ["PpeScaleShift", "PpeScaleRound", "PpeScaleMult"]>;

// ppe_scale_ctrl
def reg_PpeScaleCtrl : VPU37XX_RegisterWrapper<"PpeScaleCtrl", 32, 0x154, ["PpeScaleOverride", "PpeFpScaleOverride"]>;

// ppe_prelu
def reg_PpePrelu : VPU37XX_RegisterWrapper<"PpePrelu", 32, 0x158, ["PpePreluShift", "PpePreluMult"]>;

// ppe_misc
def reg_PpeMisc : VPU37XX_RegisterWrapper<"PpeMisc", 32, 0x170, ["PpeFp16Ftz", "PpeFp16Clamp", "PpeI32Convert"]>;

// ppe_fp_cfg
def reg_PpeFpCfg : VPU37XX_RegisterWrapper<"PpeFpCfg", 32, 0x180, ["PpeFpConvert", "PpeFpBypass", "PpeBf16Round", "PpeFpPreluEn"]>;

//
// DPUVariantRegisters
//

// workload_size0
def reg_WorkloadSize0 : VPU37XX_RegisterWrapper<"WorkloadSize0", 32, 0x20, ["WorkloadSizeX", "WorkloadSizeY"]>;

// workload_size1
def reg_WorkloadSize1 : VPU37XX_RegisterWrapper<"WorkloadSize1", 32, 0x24,
    ["WorkloadSizeZ", "PadCountUp", "PadCountLeft", "PadCountDown", "PadCountRight"]>;

// workload_start0
def reg_WorkloadStart0 : VPU37XX_RegisterWrapper<"WorkloadStart0", 32, 0x28, ["WorkloadStartX", "WorkloadStartY"]>;

// workload_start1
def reg_WorkloadStart1 : VPU37XX_RegisterWrapper<"WorkloadStart1", 32, 0x2c, ["WorkloadStartZ"]>;

// offset_addr
def reg_OffsetAddr : VPU37XX_RegisterWrapper<"OffsetAddr", 32, 0x30,
    ["NthwNtk", "BinCfg", "ConvCond", "DenseSe", "IdxQuad", "SwizzleKeyOffsetAddr", "IduMrmClkEn", "OduClkEn",
    "MpeClkEn", "PpeClkEn", "OduStatEn", "IduStatEn", "OduStatClrMode", "IduStatClrMode", "ShaveL2CacheEn",
    "IduDbgEn", "WtSwizzleKey", "WtSwizzleSel"]>;

// te_end0
def reg_TeEnd0 : VPU37XX_RegisterWrapper<"TeEnd0", 32, 0x34, ["TeEndY", "TeEndZ"]>;

// te_end1
def reg_TeEnd1 : VPU37XX_RegisterWrapper<"TeEnd1", 32, 0x38, ["TeEndX"]>;

// te_beg0
def reg_TeBeg0 : VPU37XX_RegisterWrapper<"TeBeg0", 32, 0x3c, ["TeBegY", "TeBegZ"]>;

// te_beg1
def reg_TeBeg1 : VPU37XX_RegisterWrapper<"TeBeg1", 32, 0x40, ["TeBegX"]>;

//
// RegMapped Specialization
//

class VPU37XX_RegMappedWrapper<string name = "", list<string> regsTypes = []> :
    VPURegMapped_RegMappedWrapper<VPU37XX_Dialect, "vpux::VPU37XX::", name, regsTypes> {

}

// DPUInvariantRegister
def reg_mapped_DpuInvariantRegister : VPU37XX_RegMappedWrapper<"DpuInvariantRegister",
    ["ZConfig", "KernelPadCfg", "TensorSize0", "TensorSize1", "TensorMode", "ElopsWload", "OduCfg", "TeDim0",
    "OduCast", "MpeCfg", "ElopScale", "PpeCfg", "PpeScale", "PpeScaleCtrl", "PpePrelu", "PpeMisc", "PpeFpCfg"]>;

// DPUVariantRegister
def reg_mapped_DpuVariantRegister : VPU37XX_RegMappedWrapper<"DpuVariantRegister",
    ["WorkloadSize0", "WorkloadSize1", "WorkloadStart0", "WorkloadStart1", "OffsetAddr", "TeEnd0", "TeEnd1", "TeBeg0", "TeBeg1"]>;

#endif

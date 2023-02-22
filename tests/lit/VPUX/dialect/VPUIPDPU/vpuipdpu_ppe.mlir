//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | FileCheck %s --match-full-lines
// REQUIRES: arch-VPUX37XX

module @Test {
  %weight_table = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16xi64, [@CMX_NN, 0]>
  %output_activations = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<16x64x64xf16, [@CMX_NN, 0]>

  // Use case #1a: u8 DPU in, u8 DPU out - with activation scaling
  VPUIPDPU.DPUInvariant
            weight_table(%weight_table: memref<16xi64, [@CMX_NN, 0]>)
            out_activations(%output_activations: memref<16x64x64xf16, [@CMX_NN, 0]>) {} {
    ^bb0(%scale_table:memref<16xi64, [@CMX_NN, 0]>, %out_activations:memref<16x64x64xf16, [@CMX_NN, 0]>):
        VPUIPDPU.IDUCfg
        VPUIPDPU.PPECfg {} {
        VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
        VPUIPDPU.PPEFpConvert convert_mode(FpConv_Bypass)
        VPUIPDPU.PPEIntBiasAdd %scale_table:memref<16xi64, [@CMX_NN, 0]>
        VPUIPDPU.PPEIntScaleMult %scale_table:memref<16xi64, [@CMX_NN, 0]>
        VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
        VPUIPDPU.PPEIntScaleShift %scale_table:memref<16xi64, [@CMX_NN, 0]>
        VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
        VPUIPDPU.PPEIntRound round_mode(IntRound_TiesToEven)
        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(-128)
        VPUIPDPU.PPEIntClamp clamp_low(0) clamp_high(255)
        VPUIPDPU.PPEIntConvert convert_mode(IntConv_Bypass)
        }
        VPUIPDPU.ODUCfg {} {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<16x64x64xf16, [@CMX_NN, 0]>)
        }
  }

  // Use case #1b: u8 DPU in, u8 DPU out - with activation truncation
  VPUIPDPU.DPUInvariant
            weight_table(%weight_table: memref<16xi64, [@CMX_NN, 0]>)
            out_activations(%output_activations: memref<16x64x64xf16, [@CMX_NN, 0]>) {} {
    ^bb0(%scale_table:memref<16xi64, [@CMX_NN, 0]>, %out_activations:memref<16x64x64xf16, [@CMX_NN, 0]>):
        VPUIPDPU.IDUCfg
        VPUIPDPU.PPECfg {} {
        VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
        VPUIPDPU.PPEFpConvert convert_mode(FpConv_Bypass)
        VPUIPDPU.PPEIntBiasAdd bias_static(0)
        VPUIPDPU.PPEIntScaleMult scale_static(1)
        VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
        VPUIPDPU.PPEIntScaleShift shift_static(0)
        VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
        VPUIPDPU.PPEIntRound round_mode(IntRound_Bypass)
        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
        VPUIPDPU.PPEIntClamp clamp_low(0) clamp_high(255)
        VPUIPDPU.PPEIntConvert convert_mode(IntConv_Bypass)
        }
        VPUIPDPU.ODUCfg {} {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<16x64x64xf16, [@CMX_NN, 0]>)
        }
  }

  // Use case #2: u8 DPU in, fp16 DPU out
  VPUIPDPU.DPUInvariant
            weight_table(%weight_table: memref<16xi64, [@CMX_NN, 0]>)
            out_activations(%output_activations: memref<16x64x64xf16, [@CMX_NN, 0]>) {} {
    ^bb0(%scale_table:memref<16xi64, [@CMX_NN, 0]>, %out_activations:memref<16x64x64xf16, [@CMX_NN, 0]>):
        VPUIPDPU.IDUCfg
        VPUIPDPU.PPECfg {} {
        VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
        VPUIPDPU.PPEFpConvert convert_mode(FpConv_Bypass)
        VPUIPDPU.PPEIntBiasAdd %scale_table:memref<16xi64, [@CMX_NN, 0]>
        VPUIPDPU.PPEIntScaleMult %scale_table:memref<16xi64, [@CMX_NN, 0]>
        VPUIPDPU.PPEIntPreluMult prelu_mult_static(0)
        VPUIPDPU.PPEIntScaleShift %scale_table:memref<16xi64, [@CMX_NN, 0]>
        VPUIPDPU.PPEIntConvert convert_mode(IntConv_Fp16_RNE)
        VPUIPDPU.PPEIntClamp clamp_high(70) // 70 corresponds to RELU6
        }
        VPUIPDPU.ODUCfg {} {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<16x64x64xf16, [@CMX_NN, 0]>)
        }
  }

  // Use case #3: fp16 DPU in, fp16 DPU out
  VPUIPDPU.DPUInvariant
            weight_table(%weight_table: memref<16xi64, [@CMX_NN, 0]>)
            out_activations(%output_activations: memref<16x64x64xf16, [@CMX_NN, 0]>) {} {
    ^bb0(%scale_table:memref<16xi64, [@CMX_NN, 0]>, %out_activations:memref<16x64x64xf16, [@CMX_NN, 0]>):
        VPUIPDPU.IDUCfg
        VPUIPDPU.PPECfg {} {
        VPUIPDPU.PPEFpBiasAdd %scale_table:memref<16xi64, [@CMX_NN, 0]>
        VPUIPDPU.PPEFpScaleMult %scale_table:memref<16xi64, [@CMX_NN, 0]> prelu_alpha(0.1)
        VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOff)
        VPUIPDPU.PPEFpConvert convert_mode(FpConv_Fp16_RNE) clamp_mode(FpConv_Clamp_On) ftz_mode(FpConv_FTZ_Off)
        VPUIPDPU.PPEIntBiasAdd bias_static(0)
        VPUIPDPU.PPEIntScaleMult scale_static(1)
        VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
        VPUIPDPU.PPEIntScaleShift shift_static(0)
        VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
        VPUIPDPU.PPEIntRound round_mode(IntRound_Bypass)
        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
        VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647) // (MIN_I32, MAX_I32)
        VPUIPDPU.PPEIntConvert convert_mode(IntConv_Bypass)
        }
        VPUIPDPU.ODUCfg {} {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<16x64x64xf16, [@CMX_NN, 0]>)
        }
  }

  // Use case #4: fp16 DPU in, u8 DPU out
  VPUIPDPU.DPUInvariant
            weight_table(%weight_table: memref<16xi64, [@CMX_NN, 0]>)
            out_activations(%output_activations: memref<16x64x64xf16, [@CMX_NN, 0]>) {} {
    ^bb0(%scale_table:memref<16xi64, [@CMX_NN, 0]>, %out_activations:memref<16x64x64xf16, [@CMX_NN, 0]>):
        VPUIPDPU.IDUCfg
        VPUIPDPU.PPECfg {} {
        VPUIPDPU.PPEFpBiasAdd %scale_table:memref<16xi64, [@CMX_NN, 0]>
        VPUIPDPU.PPEFpScaleMult %scale_table:memref<16xi64, [@CMX_NN, 0]> prelu_alpha(0.1)
        VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOff)
        VPUIPDPU.PPEFpConvert convert_mode(FpConv_I32_RNE)
        VPUIPDPU.PPEIntBiasAdd bias_static(0)
        VPUIPDPU.PPEIntScaleMult scale_static(1)
        VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
        VPUIPDPU.PPEIntScaleShift shift_static(0)
        VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
        VPUIPDPU.PPEIntRound round_mode(IntRound_Bypass)
        VPUIPDPU.PPEIntZeroPointOffset zero_point_static(-128)
        VPUIPDPU.PPEIntClamp clamp_low(0) clamp_high(255)
        VPUIPDPU.PPEIntConvert convert_mode(IntConv_Bypass)
        }
        VPUIPDPU.ODUCfg {} {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<16x64x64xf16, [@CMX_NN, 0]>)
        }
  }
}

// CHECK:   module @Test attributes {VPU.arch = "VPUX{{(37)|(40)}}XX"} {
// CHECK:     %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16xi64, [@CMX_NN, 0]>
// CHECK:     %1 = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<16x64x64xf16, [@CMX_NN, 0]>
// CHECK:     %2 = VPUIPDPU.DPUInvariant weight_table(%0 : memref<16xi64, [@CMX_NN, 0]>) out_activations(%1 : memref<16x64x64xf16, [@CMX_NN, 0]>) {
// CHECK:     ^bb0(%arg0: memref<16xi64, [@CMX_NN, 0]>, %arg1: memref<16x64x64xf16, [@CMX_NN, 0]>):
// CHECK:       VPUIPDPU.IDUCfg
// CHECK:       VPUIPDPU.PPECfg {
// CHECK:         VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
// CHECK:         VPUIPDPU.PPEFpConvert convert_mode(FpConv_Bypass)
// CHECK:         VPUIPDPU.PPEIntBiasAdd %arg0 : memref<16xi64, [@CMX_NN, 0]>
// CHECK:         VPUIPDPU.PPEIntScaleMult %arg0 : memref<16xi64, [@CMX_NN, 0]>
// CHECK:         VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
// CHECK:         VPUIPDPU.PPEIntScaleShift %arg0 : memref<16xi64, [@CMX_NN, 0]>
// CHECK:         VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
// CHECK:         VPUIPDPU.PPEIntRound round_mode(IntRound_TiesToEven)
// CHECK:         VPUIPDPU.PPEIntZeroPointOffset zero_point_static(-128)
// CHECK:         VPUIPDPU.PPEIntClamp clamp_low(0) clamp_high(255)
// CHECK:         VPUIPDPU.PPEIntConvert convert_mode(IntConv_Bypass)
// CHECK:       }
// CHECK:       VPUIPDPU.ODUCfg {
// CHECK:         VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
// CHECK:         VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<16x64x64xf16, [@CMX_NN, 0]>)
// CHECK:       }
// CHECK:     }
// CHECK:     %3 = VPUIPDPU.DPUInvariant weight_table(%0 : memref<16xi64, [@CMX_NN, 0]>) out_activations(%1 : memref<16x64x64xf16, [@CMX_NN, 0]>) {
// CHECK:     ^bb0(%arg0: memref<16xi64, [@CMX_NN, 0]>, %arg1: memref<16x64x64xf16, [@CMX_NN, 0]>):
// CHECK:       VPUIPDPU.IDUCfg
// CHECK:       VPUIPDPU.PPECfg {
// CHECK:         VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
// CHECK:         VPUIPDPU.PPEFpConvert convert_mode(FpConv_Bypass)
// CHECK:         VPUIPDPU.PPEIntBiasAdd bias_static(0)
// CHECK:         VPUIPDPU.PPEIntScaleMult scale_static(1)
// CHECK:         VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
// CHECK:         VPUIPDPU.PPEIntScaleShift shift_static(0)
// CHECK:         VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
// CHECK:         VPUIPDPU.PPEIntRound round_mode(IntRound_Bypass)
// CHECK:         VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
// CHECK:         VPUIPDPU.PPEIntClamp clamp_low(0) clamp_high(255)
// CHECK:         VPUIPDPU.PPEIntConvert convert_mode(IntConv_Bypass)
// CHECK:       }
// CHECK:       VPUIPDPU.ODUCfg {
// CHECK:         VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
// CHECK:         VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<16x64x64xf16, [@CMX_NN, 0]>)
// CHECK:       }
// CHECK:     }
// CHECK:     %4 = VPUIPDPU.DPUInvariant weight_table(%0 : memref<16xi64, [@CMX_NN, 0]>) out_activations(%1 : memref<16x64x64xf16, [@CMX_NN, 0]>) {
// CHECK:     ^bb0(%arg0: memref<16xi64, [@CMX_NN, 0]>, %arg1: memref<16x64x64xf16, [@CMX_NN, 0]>):
// CHECK:       VPUIPDPU.IDUCfg
// CHECK:       VPUIPDPU.PPECfg {
// CHECK:         VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
// CHECK:         VPUIPDPU.PPEFpConvert convert_mode(FpConv_Bypass)
// CHECK:         VPUIPDPU.PPEIntBiasAdd %arg0 : memref<16xi64, [@CMX_NN, 0]>
// CHECK:         VPUIPDPU.PPEIntScaleMult %arg0 : memref<16xi64, [@CMX_NN, 0]>
// CHECK:         VPUIPDPU.PPEIntPreluMult prelu_mult_static(0)
// CHECK:         VPUIPDPU.PPEIntScaleShift %arg0 : memref<16xi64, [@CMX_NN, 0]>
// CHECK:         VPUIPDPU.PPEIntConvert convert_mode(IntConv_Fp16_RNE)
// CHECK:         VPUIPDPU.PPEIntClamp clamp_high(70)
// CHECK:       }
// CHECK:       VPUIPDPU.ODUCfg {
// CHECK:         VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
// CHECK:         VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<16x64x64xf16, [@CMX_NN, 0]>)
// CHECK:       }
// CHECK:     }
// CHECK:     %5 = VPUIPDPU.DPUInvariant weight_table(%0 : memref<16xi64, [@CMX_NN, 0]>) out_activations(%1 : memref<16x64x64xf16, [@CMX_NN, 0]>) {
// CHECK:     ^bb0(%arg0: memref<16xi64, [@CMX_NN, 0]>, %arg1: memref<16x64x64xf16, [@CMX_NN, 0]>):
// CHECK:       VPUIPDPU.IDUCfg
// CHECK:       VPUIPDPU.PPECfg {
// CHECK:         VPUIPDPU.PPEFpBiasAdd %arg0 : memref<16xi64, [@CMX_NN, 0]>
// CHECK:         VPUIPDPU.PPEFpScaleMult %arg0 : memref<16xi64, [@CMX_NN, 0]> prelu_alpha(1.000000e-01)
// CHECK:         VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOff)
// CHECK:         VPUIPDPU.PPEFpConvert convert_mode(FpConv_Fp16_RNE) clamp_mode(FpConv_Clamp_On) ftz_mode(FpConv_FTZ_Off)
// CHECK:         VPUIPDPU.PPEIntBiasAdd bias_static(0)
// CHECK:         VPUIPDPU.PPEIntScaleMult scale_static(1)
// CHECK:         VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
// CHECK:         VPUIPDPU.PPEIntScaleShift shift_static(0)
// CHECK:         VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
// CHECK:         VPUIPDPU.PPEIntRound round_mode(IntRound_Bypass)
// CHECK:         VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
// CHECK:         VPUIPDPU.PPEIntClamp clamp_low(-2147483648) clamp_high(2147483647)
// CHECK:         VPUIPDPU.PPEIntConvert convert_mode(IntConv_Bypass)
// CHECK:       }
// CHECK:       VPUIPDPU.ODUCfg {
// CHECK:         VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
// CHECK:         VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<16x64x64xf16, [@CMX_NN, 0]>)
// CHECK:       }
// CHECK:     }
// CHECK:     %6 = VPUIPDPU.DPUInvariant weight_table(%0 : memref<16xi64, [@CMX_NN, 0]>) out_activations(%1 : memref<16x64x64xf16, [@CMX_NN, 0]>) {
// CHECK:     ^bb0(%arg0: memref<16xi64, [@CMX_NN, 0]>, %arg1: memref<16x64x64xf16, [@CMX_NN, 0]>):
// CHECK:       VPUIPDPU.IDUCfg
// CHECK:       VPUIPDPU.PPECfg {
// CHECK:         VPUIPDPU.PPEFpBiasAdd %arg0 : memref<16xi64, [@CMX_NN, 0]>
// CHECK:         VPUIPDPU.PPEFpScaleMult %arg0 : memref<16xi64, [@CMX_NN, 0]> prelu_alpha(1.000000e-01)
// CHECK:         VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOff)
// CHECK:         VPUIPDPU.PPEFpConvert convert_mode(FpConv_I32_RNE)
// CHECK:         VPUIPDPU.PPEIntBiasAdd bias_static(0)
// CHECK:         VPUIPDPU.PPEIntScaleMult scale_static(1)
// CHECK:         VPUIPDPU.PPEIntPreluMult prelu_mult_static(1)
// CHECK:         VPUIPDPU.PPEIntScaleShift shift_static(0)
// CHECK:         VPUIPDPU.PPEIntPreluShift prelu_shift_static(0)
// CHECK:         VPUIPDPU.PPEIntRound round_mode(IntRound_Bypass)
// CHECK:         VPUIPDPU.PPEIntZeroPointOffset zero_point_static(-128)
// CHECK:         VPUIPDPU.PPEIntClamp clamp_low(0) clamp_high(255)
// CHECK:         VPUIPDPU.PPEIntConvert convert_mode(IntConv_Bypass)
// CHECK:       }
// CHECK:       VPUIPDPU.ODUCfg {
// CHECK:         VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
// CHECK:         VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<16x64x64xf16, [@CMX_NN, 0]>)
// CHECK:       }
// CHECK:     }
// CHECK:   }

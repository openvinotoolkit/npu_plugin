//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | FileCheck %s --match-full-lines
// REQUIRES: arch-VPUX37XX

module @Test {
  %weight_table = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16xi64, [@CMX_NN, 0]>
  %weight_table_i8 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16xi8, [@CMX_NN, 0]>
  %output_activations = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<16x64x64xf16, [@CMX_NN, 0]>

  // Use case #1:
  VPUIPDPU.DPUInvariant
            weight_table(%weight_table: memref<16xi64, [@CMX_NN, 0]>)
            out_activations(%output_activations: memref<16x64x64xf16, [@CMX_NN, 0]>) {} {
    ^bb0(%scale_table:memref<16xi64, [@CMX_NN, 0]>, %out_activations:memref<16x64x64xf16, [@CMX_NN, 0]>):
        VPUIPDPU.IDUCfg
// Note1: After IDU update, make sure DTZ is only used on f16 input
        VPUIPDPU.MPECfg {} {
            VPUIPDPU.MPEDenormalOperandsFTZ
        }
        VPUIPDPU.PPECfg {} {
        VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
        }
        VPUIPDPU.ODUCfg {} {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<16x64x64xf16, [@CMX_NN, 0]>)
        }
  }

  // Use case #2:
  VPUIPDPU.DPUInvariant
            weight_table(%weight_table: memref<16xi64, [@CMX_NN, 0]>)
            out_activations(%output_activations: memref<16x64x64xf16, [@CMX_NN, 0]>) {} {
    ^bb0(%scale_table:memref<16xi64, [@CMX_NN, 0]>, %out_activations:memref<16x64x64xf16, [@CMX_NN, 0]>):
        VPUIPDPU.IDUCfg
        VPUIPDPU.MPECfg {} {
// Note2: The bias needs to be I8 for I8 input and U8 for U8 input. Check will be added once IDU is done.
            VPUIPDPU.MPEActivationBias act_bias(-12)
        }
        VPUIPDPU.PPECfg {} {
        VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
        }
        VPUIPDPU.ODUCfg {} {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<16x64x64xf16, [@CMX_NN, 0]>)
        }
  }

  // Use case #3:
  VPUIPDPU.DPUInvariant
            weight_table(%weight_table_i8: memref<16xi8, [@CMX_NN, 0]>)
            out_activations(%output_activations: memref<16x64x64xf16, [@CMX_NN, 0]>) {} {
    ^bb0(%scale_table:memref<16xi64, [@CMX_NN, 0]>, %out_activations:memref<16x64x64xf16, [@CMX_NN, 0]>):
        VPUIPDPU.IDUCfg
        VPUIPDPU.MPECfg {} {
            VPUIPDPU.MPEWeightsBias weights_bias(-10)
        }
        VPUIPDPU.PPECfg {} {
        VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
        }
        VPUIPDPU.ODUCfg {} {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<16x64x64xf16, [@CMX_NN, 0]>)
        }
  }

}

// CHECK:   module @Test attributes {VPU.arch = "VPUX{{(37)|(40)}}XX"} {
// CHECK:     %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16xi64, [@CMX_NN, 0]>
// CHECK:     %1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16xi8, [@CMX_NN, 0]>
// CHECK:     %2 = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<16x64x64xf16, [@CMX_NN, 0]>
// CHECK:     %3 = VPUIPDPU.DPUInvariant weight_table(%0 : memref<16xi64, [@CMX_NN, 0]>) out_activations(%2 : memref<16x64x64xf16, [@CMX_NN, 0]>) {
// CHECK:     ^bb0(%arg0: memref<16xi64, [@CMX_NN, 0]>, %arg1: memref<16x64x64xf16, [@CMX_NN, 0]>):
// CHECK:       VPUIPDPU.IDUCfg
// CHECK:       VPUIPDPU.MPECfg {
// CHECK:           VPUIPDPU.MPEDenormalOperandsFTZ
// CHECK:       }
// CHECK:       VPUIPDPU.PPECfg {
// CHECK:         VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
// CHECK:       }
// CHECK:       VPUIPDPU.ODUCfg {
// CHECK:         VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
// CHECK:         VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<16x64x64xf16, [@CMX_NN, 0]>)
// CHECK:       }
// CHECK:     }
// CHECK:     %4 = VPUIPDPU.DPUInvariant weight_table(%0 : memref<16xi64, [@CMX_NN, 0]>) out_activations(%2 : memref<16x64x64xf16, [@CMX_NN, 0]>) {
// CHECK:     ^bb0(%arg0: memref<16xi64, [@CMX_NN, 0]>, %arg1: memref<16x64x64xf16, [@CMX_NN, 0]>):
// CHECK:       VPUIPDPU.IDUCfg
// CHECK:       VPUIPDPU.MPECfg {
// CHECK:           VPUIPDPU.MPEActivationBias act_bias(-12)
// CHECK:       }
// CHECK:       VPUIPDPU.PPECfg {
// CHECK:         VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
// CHECK:       }
// CHECK:       VPUIPDPU.ODUCfg {
// CHECK:         VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
// CHECK:         VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<16x64x64xf16, [@CMX_NN, 0]>)
// CHECK:       }
// CHECK:     }
// CHECK:     %5 = VPUIPDPU.DPUInvariant weight_table(%1 : memref<16xi8, [@CMX_NN, 0]>) out_activations(%2 : memref<16x64x64xf16, [@CMX_NN, 0]>) {
// CHECK:     ^bb0(%arg0: memref<16xi64, [@CMX_NN, 0]>, %arg1: memref<16x64x64xf16, [@CMX_NN, 0]>):
// CHECK:       VPUIPDPU.IDUCfg
// CHECK:       VPUIPDPU.MPECfg {
// CHECK:           VPUIPDPU.MPEWeightsBias weights_bias(-10)
// CHECK:       }
// CHECK:       VPUIPDPU.PPECfg {
// CHECK:         VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
// CHECK:       }
// CHECK:       VPUIPDPU.ODUCfg {
// CHECK:         VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
// CHECK:         VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<16x64x64xf16, [@CMX_NN, 0]>)
// CHECK:       }
// CHECK:     }
// CHECK:   }

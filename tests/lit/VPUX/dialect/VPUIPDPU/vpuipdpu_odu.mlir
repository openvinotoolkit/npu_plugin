// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | FileCheck %s --match-full-lines
// REQUIRES: arch-VPUX37XX

module @Test {
  %weight_table = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16xi64, [@CMX_NN, 0]>
  %output_activations = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<16x64x64xf16, [@CMX_NN, 0]>
  %output_sparsity = VPURT.DeclareBuffer "CMX_NN" [0] <131200> -> memref<16x64x64xi1, [@CMX_NN, 0]>

  // Scenario#1
  %invariant0 = VPUIPDPU.DPUInvariant
                    weight_table(%weight_table: memref<16xi64, [@CMX_NN, 0]>)
                    out_activations(%output_activations: memref<16x64x64xf16, [@CMX_NN, 0]>)
                    sparsity_map(%output_sparsity: memref<16x64x64xi1, [@CMX_NN, 0]>) {} {
    ^bb0(%scale_table:memref<16xi64, [@CMX_NN, 0]>, %out_activations:memref<16x64x64xf16, [@CMX_NN, 0]>, %sparsity_map: memref<16x64x64xi1, [@CMX_NN, 0]>):
        VPUIPDPU.IDUCfg
        VPUIPDPU.MPECfg
        VPUIPDPU.PPECfg {} {
            VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
        }
        VPUIPDPU.ODUCfg {} {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
            VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZXY)
            VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
            VPUIPDPU.ODUSparsity %sparsity_map: memref<16x64x64xi1, [@CMX_NN, 0]>
            VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<16x64x64xf16, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
            VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
        }
  }

  VPUIPDPU.DPUVariant invariant(%invariant0) {} {
    VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
    VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(3) end_coord_y(3)
                            activations_offset(0) sparsity_offset(0) target_width(0)
                            cast_to_tile("DPU_TILE_1|DPU_TILE_2")
    VPUIPDPU.ODUHaloRegion begin_coord_x(60) begin_coord_y(60) end_coord_x(63) end_coord_y(63)
                            activations_offset(0) sparsity_offset(0) target_width(0)
                            cast_to_tile("DPU_TILE_1|DPU_TILE_2")
  }

  VPUIPDPU.DPUVariant invariant(%invariant0) {} {
    VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
    VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x (63) end_coord_y(63)
                            activations_offset(0) sparsity_offset(0) target_width(0)
                            cast_to_tile("DPU_TILE_2|DPU_TILE_3")
  }

  // Scenario#2
  %invariant1 = VPUIPDPU.DPUInvariant
                    weight_table(%weight_table: memref<16xi64, [@CMX_NN, 0]>)
                    out_activations(%output_activations: memref<16x64x64xf16, [@CMX_NN, 0]>)
                    sparsity_map(%output_sparsity: memref<16x64x64xi1, [@CMX_NN, 0]>) {} {
    ^bb0(%scale_table:memref<16xi64, [@CMX_NN, 0]>, %out_activations:memref<16x64x64xf16, [@CMX_NN, 0]>, %sparsity_map: memref<16x64x64xi1, [@CMX_NN, 0]>):
        VPUIPDPU.IDUCfg
        VPUIPDPU.MPECfg
        VPUIPDPU.PPECfg {} {
            VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
        }
        VPUIPDPU.ODUCfg {} {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZXY)
            VPUIPDPU.ODUSparsity compression_enabled(true) sparse_value(0)
            VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<16x64x64xf16, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
            VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS)
        }
  }

  VPUIPDPU.DPUVariant invariant(%invariant1) {} {
      VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
  }

  // Scenario#3
  %invariant2 = VPUIPDPU.DPUInvariant
                    weight_table(%weight_table: memref<16xi64, [@CMX_NN, 0]>)
                    out_activations(%output_activations: memref<16x64x64xf16, [@CMX_NN, 0]>)
                    sparsity_map(%output_sparsity: memref<16x64x64xi1, [@CMX_NN, 0]>) {} {
    ^bb0(%scale_table:memref<16xi64, [@CMX_NN, 0]>, %out_activations:memref<16x64x64xf16, [@CMX_NN, 0]>, %sparsity_map: memref<16x64x64xi1, [@CMX_NN, 0]>):
        VPUIPDPU.IDUCfg
        VPUIPDPU.MPECfg
        VPUIPDPU.PPECfg {} {
            VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
        }
        VPUIPDPU.ODUCfg {} {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
            VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_YXZ)
            VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
            VPUIPDPU.ODUSparsity %sparsity_map: memref<16x64x64xi1, [@CMX_NN, 0]> sparse_value(0)
            VPUIPDPU.ODUOutActivations out_activations(%out_activations: memref<16x64x64xf16, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
            VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
        }
  }

  VPUIPDPU.DPUVariant invariant(%invariant2) {} {
    VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
    VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(4) end_coord_y(4)
                            activations_offset(0) sparsity_offset(0) target_width(0)
                            cast_to_tile("DPU_TILE_1|DPU_TILE_2")
    VPUIPDPU.ODUHaloRegion begin_coord_x(59) begin_coord_y(59) end_coord_x(63) end_coord_y(63)
                            activations_offset(0) sparsity_offset(0) target_width(0)
                            cast_to_tile("DPU_TILE_1|DPU_TILE_2")
    VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(63) end_coord_y(63)
                            activations_offset(0) sparsity_offset(0) target_width(0)
                            cast_to_tile("DPU_TILE_3|DPU_TILE_4")
    VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(100) end_coord_y(100)
                            activations_offset(131072) sparsity_offset(0) target_width(0)
                            cast_to_tile("DPU_TILE_1|DPU_TILE_2|DPU_TILE_3")
    VPUIPDPU.ODUHaloRegion begin_coord_x(29) begin_coord_y(29) end_coord_x(35) end_coord_y(35)
                            activations_offset(262144) sparsity_offset(0) target_width(0)
                            cast_to_tile("DPU_TILE_1|DPU_TILE_2|DPU_TILE_3|DPU_TILE_4")
  }
}

// CHECK:   module @Test attributes {VPU.arch = "VPUX{{(37)|(40)}}XX"} {
// CHECK:     %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16xi64, [@CMX_NN, 0]>
// CHECK:     %1 = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<16x64x64xf16, [@CMX_NN, 0]>
// CHECK:     %2 = VPURT.DeclareBuffer "CMX_NN" [0] <131200> -> memref<16x64x64xi1, [@CMX_NN, 0]>
// CHECK:     %3 = VPUIPDPU.DPUInvariant weight_table(%0 : memref<16xi64, [@CMX_NN, 0]>) out_activations(%1 : memref<16x64x64xf16, [@CMX_NN, 0]>) sparsity_map(%2 : memref<16x64x64xi1, [@CMX_NN, 0]>) {
// CHECK:     ^bb0(%arg0: memref<16xi64, [@CMX_NN, 0]>, %arg1: memref<16x64x64xf16, [@CMX_NN, 0]>, %arg2: memref<16x64x64xi1, [@CMX_NN, 0]>):
// CHECK:       VPUIPDPU.IDUCfg
// CHECK:       VPUIPDPU.MPECfg
// CHECK:       VPUIPDPU.PPECfg {
// CHECK:         VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
// CHECK:       }
// CHECK:       VPUIPDPU.ODUCfg {
// CHECK:         VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
// CHECK:         VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
// CHECK:         VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZXY)
// CHECK:         VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
// CHECK:         VPUIPDPU.ODUSparsity %arg2 : memref<16x64x64xi1, [@CMX_NN, 0]>
// CHECK:         VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<16x64x64xf16, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
// CHECK:         VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
// CHECK:       }
// CHECK:     }
// CHECK:     VPUIPDPU.DPUVariant invariant(%3) {
// CHECK:       VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
// CHECK:       VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(3) end_coord_y(3) activations_offset(0) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_1|DPU_TILE_2)
// CHECK:       VPUIPDPU.ODUHaloRegion begin_coord_x(60) begin_coord_y(60) end_coord_x(63) end_coord_y(63) activations_offset(0) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_1|DPU_TILE_2)
// CHECK:     }
// CHECK:     VPUIPDPU.DPUVariant invariant(%3) {
// CHECK:       VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
// CHECK:       VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(63) end_coord_y(63) activations_offset(0) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_2|DPU_TILE_3)
// CHECK:     }
// CHECK:     %4 = VPUIPDPU.DPUInvariant weight_table(%0 : memref<16xi64, [@CMX_NN, 0]>) out_activations(%1 : memref<16x64x64xf16, [@CMX_NN, 0]>) sparsity_map(%2 : memref<16x64x64xi1, [@CMX_NN, 0]>) {
// CHECK:     ^bb0(%arg0: memref<16xi64, [@CMX_NN, 0]>, %arg1: memref<16x64x64xf16, [@CMX_NN, 0]>, %arg2: memref<16x64x64xi1, [@CMX_NN, 0]>):
// CHECK:       VPUIPDPU.IDUCfg
// CHECK:       VPUIPDPU.MPECfg
// CHECK:       VPUIPDPU.PPECfg {
// CHECK:         VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
// CHECK:       }
// CHECK:       VPUIPDPU.ODUCfg {
// CHECK:         VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
// CHECK:         VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZXY)
// CHECK:         VPUIPDPU.ODUSparsity compression_enabled(true) sparse_value(0)
// CHECK:         VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<16x64x64xf16, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
// CHECK:         VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS)
// CHECK:       }
// CHECK:     }
// CHECK:     VPUIPDPU.DPUVariant invariant(%4) {
// CHECK:       VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
// CHECK:     }
// CHECK:     %5 = VPUIPDPU.DPUInvariant weight_table(%0 : memref<16xi64, [@CMX_NN, 0]>) out_activations(%1 : memref<16x64x64xf16, [@CMX_NN, 0]>) sparsity_map(%2 : memref<16x64x64xi1, [@CMX_NN, 0]>) {
// CHECK:     ^bb0(%arg0: memref<16xi64, [@CMX_NN, 0]>, %arg1: memref<16x64x64xf16, [@CMX_NN, 0]>, %arg2: memref<16x64x64xi1, [@CMX_NN, 0]>):
// CHECK:       VPUIPDPU.IDUCfg
// CHECK:       VPUIPDPU.MPECfg
// CHECK:       VPUIPDPU.PPECfg {
// CHECK:         VPUIPDPU.PPEFpAddMultBypass bypass_mode(BypassOn)
// CHECK:       }
// CHECK:       VPUIPDPU.ODUCfg {
// CHECK:         VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
// CHECK:         VPUIPDPU.ODUDataReuse activation_reuse(NTHW_16)
// CHECK:         VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_YXZ)
// CHECK:         VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
// CHECK:         VPUIPDPU.ODUSparsity %arg2 : memref<16x64x64xi1, [@CMX_NN, 0]> sparse_value(0)
// CHECK:         VPUIPDPU.ODUOutActivations out_activations(%arg1 : memref<16x64x64xf16, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
// CHECK:         VPUIPDPU.ODUWriteCombineBuffer activations_mode(WCB_COMBINE_BY_ADDRESS) sparsity_mode(WCB_COMBINE_BY_CONTEXT)
// CHECK:       }
// CHECK:     }
// CHECK:     VPUIPDPU.DPUVariant invariant(%5) {
// CHECK:       VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
// CHECK:       VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(4) end_coord_y(4) activations_offset(0) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_1|DPU_TILE_2)
// CHECK:       VPUIPDPU.ODUHaloRegion begin_coord_x(59) begin_coord_y(59) end_coord_x(63) end_coord_y(63) activations_offset(0) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_1|DPU_TILE_2)
// CHECK:       VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(63) end_coord_y(63) activations_offset(0) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_3|DPU_TILE_4)
// CHECK:       VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(100) end_coord_y(100) activations_offset(131072) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_1|DPU_TILE_2|DPU_TILE_3)
// CHECK:       VPUIPDPU.ODUHaloRegion begin_coord_x(29) begin_coord_y(29) end_coord_x(35) end_coord_y(35) activations_offset(262144) sparsity_offset(0) target_width(0) cast_to_tile(DPU_TILE_1|DPU_TILE_2|DPU_TILE_3|DPU_TILE_4)
// CHECK:     }
// CHECK:   }

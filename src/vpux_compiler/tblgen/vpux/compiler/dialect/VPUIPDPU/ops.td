//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU_OPS
#define VPUX_COMPILER_DIALECT_VPUIPDPU_OPS

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/core/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIPDPU/attributes.td"
include "vpux/compiler/dialect/VPUIPDPU/dialect.td"
include "vpux/compiler/dialect/VPUIPDPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIPDPU/types.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"

//
// Base classes
//

class VPUIPDPU_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPUIPDPU_Dialect,
            mnemonic,
            traits
        >;

//
// DPU Invariant
//

def VPUIPDPU_DPUInvariant :
        VPUIPDPU_Op<"DPUInvariant",
            [
                NoTerminator
            ]
        > {
    let summary = "DPU Invariant configuration";

    let arguments = (ins
        AnyMemRef:$weight_table,
        AnyMemRef:$out_activations,
        Optional<AnyMemRef>:$sparsity_map
    );

    let results = (outs VPUIPDPU_Invariant:$invariant);

    let regions = (region
        SizedRegion<1>:$region
    );
    let assemblyFormat = [{
        `weight_table` `(` $weight_table `:` type($weight_table) `)`
        `out_activations` `(` $out_activations `:` type($out_activations) `)`
        (`sparsity_map` `(` $sparsity_map^ `:` type($sparsity_map) `)`)?
        attr-dict
        $region
    }];

    let verifier = [{
        return vpux::VPUIPDPU::verifyOp(*this);
    }];
}

//
// DPU Variant
//

def VPUIPDPU_DPUVariant :
        VPUIPDPU_Op<"DPUVariant",
            [
                NoTerminator
            ]
        > {
    let summary = "DPU Variant configuration";

    let arguments = (ins
        VPUIPDPU_Invariant:$invariant
    );

    let regions = (region
        SizedRegion<1>:$region
    );

    let assemblyFormat = [{
        `invariant` `(` $invariant `)`
        attr-dict
        $region
    }];

    let verifier = [{
        return vpux::VPUIPDPU::verifyOp(*this);
    }];
}

//
// IDUCfg
//

def VPUIPDPU_IDUCfgOp :
        VPUIPDPU_Op<"IDUCfg",
            [
                HasParent<"vpux::VPUIPDPU::DPUInvariant">
            ]
        > {
    let summary = "IDU configuration";

    let assemblyFormat = [{
        attr-dict
    }];
}

//
// MPECfg
//

def VPUIPDPU_MPECfgOp :
        VPUIPDPU_Op<"MPECfg",
            [
                HasParent<"vpux::VPUIPDPU::DPUInvariant">,
                NoTerminator
            ]
        > {
    let summary = "MPE configuration";

    let regions = (region
        SizedRegion<1>:$region
    );

    let assemblyFormat = [{
        attr-dict
        $region
    }];

    let verifier = [{
        return vpux::VPUIPDPU::verifyOp(*this);
    }];
}

//
// PPECfg
//

def VPUIPDPU_PPECfgOp :
        VPUIPDPU_Op<"PPECfg",
            [
                HasParent<"vpux::VPUIPDPU::DPUInvariant">,
                NoTerminator
            ]
        > {
    let summary = "PPE configuration";

    let regions = (region
        SizedRegion<1>:$region
    );

    let assemblyFormat = [{
        attr-dict
        $region
    }];
}

//
// PPEFpBiasAddOp
//

def VPUIPDPU_PPEFpBiasAddOp :
        VPUIPDPU_Op<"PPEFpBiasAdd",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE floating point pipeline bias addition";

    let arguments = (ins
        Optional<AnyMemRef>:$scale_table,
        OptionalAttr<F32Attr>:$bias_static
    );

    let assemblyFormat = [{
        ($scale_table^ `:` type($scale_table))?
        (`bias_static` `(` $bias_static^ `)`)?
        attr-dict
    }];

    let verifier = [{
        return vpux::VPUIPDPU::verifyOp(*this);
    }];
}

//
// PPEFpScaleMultOp
//

def VPUIPDPU_PPEFpScaleMultOp :
        VPUIPDPU_Op<"PPEFpScaleMult",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE floating point pipeline scale multiplication";

    let arguments = (ins
        Optional<AnyMemRef>:$scale_table,
        OptionalAttr<F32Attr>:$scale_static,
        OptionalAttr<F32Attr>:$prelu_alpha
    );

    let assemblyFormat = [{
        ($scale_table^ `:` type($scale_table))?
        (`scale_static` `(` $scale_static^ `)`)?
        (`prelu_alpha` `(` $prelu_alpha^ `)`)?
        attr-dict
    }];

    let verifier = [{
        return vpux::VPUIPDPU::verifyOp(*this);
    }];
}

//
// PPEFpAddMultBypassOp
//

def VPUIPDPU_PPEFpAddMultBypassOp :
        VPUIPDPU_Op<"PPEFpAddMultBypass",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE floating point pipeline bypass add and mult steps";

    let arguments = (ins
        VPUIPDPU_PPEBypassMode:$bypass_mode
    );

    let assemblyFormat = [{
        `bypass_mode` `(` $bypass_mode `)`
        attr-dict
    }];
}

//
// PPEFpConvertOp
//

def VPUIPDPU_PPEFpConvertOp :
        VPUIPDPU_Op<"PPEFpConvert",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE floating point pipeline conversion";

    let description = [{
Conversion modes and attributes <-> reg. fields relation:
| Convert fp32  to | Round mode     | Reg. for round | Clamp mode                  | Ftz mode                              | Reg. for clamp/ftz            |
|:-----------------|:---------------|:---------------|:----------------------------|:--------------------------------------|:------------------------------|
| none             | N/A            | N/A            | N/A                         | N/A                                   | N/A                           |
| fp16             | rne (implicit) | N/A            | no_clamp / clamp_to_min_max | support_denoms / force_denoms_to_zero | ppe_fp16_clamp / ppe_fp16_ftz |
| bf16             | trunc / rne    | ppe_bf16_round | N/A                         | N/A                                   | N/A                           |
| fp8              | rne (implicit) | N/A            | no_clamp / clamp_to_min_max | support_denoms / force_denoms_to_zero | ppe_fp16_clamp / ppe_fp16_ftz |
| i32              | rne (implicit) | N/A            | N/A                         | N/A                                   | N/A                           |
        }];

    let arguments = (ins
        VPUIPDPU_PPEFpConvertMode:$convert_mode,
        OptionalAttr<VPUIPDPU_PPEFpConvClampMode>:$clamp_mode,
        OptionalAttr<VPUIPDPU_PPEFpConvFTZMode>:$ftz_mode,
        OptionalAttr<VPUIPDPU_PPEFpConvBf16RoundMode>:$bf16_round_mode
    );

    let assemblyFormat = [{
        `convert_mode` `(` $convert_mode `)`
        (`clamp_mode` `(` $clamp_mode^ `)`)?
        (`ftz_mode` `(` $ftz_mode^ `)`)?
        (`bf16_round_mode` `(` $bf16_round_mode^ `)`)?
        attr-dict
    }];

    let verifier = [{
        return vpux::VPUIPDPU::verifyOp(*this);
    }];
}

//
// PPEIntBiasAddOp
//

def VPUIPDPU_PPEIntBiasAddOp :
        VPUIPDPU_Op<"PPEIntBiasAdd",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline bias addition";

    let arguments = (ins
        Optional<AnyMemRef>:$scale_table,
        OptionalAttr<IntAttr>:$bias_static
    );

    let assemblyFormat = [{
        ($scale_table^ `:` type($scale_table))?
        (`bias_static` `(` $bias_static^ `)`)?
        attr-dict
    }];

    let verifier = [{
        return vpux::VPUIPDPU::verifyOp(*this);
    }];
}

//
// PPEIntScaleMultOp
//

def VPUIPDPU_PPEIntScaleMultOp :
        VPUIPDPU_Op<"PPEIntScaleMult",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline scale multiplication";

    let arguments = (ins
        Optional<AnyMemRef>:$scale_table,
        OptionalAttr<IntAttr>:$scale_static
    );

    let assemblyFormat = [{
        ($scale_table^ `:` type($scale_table))?
        (`scale_static` `(` $scale_static^ `)`)?
        attr-dict
    }];

    let verifier = [{
        return vpux::VPUIPDPU::verifyOp(*this);
    }];
}

//
// PPEIntPreluMultOp
//

def VPUIPDPU_PPEIntPreluMultOp :
        VPUIPDPU_Op<"PPEIntPreluMult",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline prelu multiplicator";

    let arguments = (ins
        IntAttr:$prelu_mult_static
    );

    let assemblyFormat = [{
        `prelu_mult_static` `(` $prelu_mult_static `)`
        attr-dict
    }];
}

//
// PPEIntScaleShiftOp
//

def VPUIPDPU_PPEIntScaleShiftOp :
        VPUIPDPU_Op<"PPEIntScaleShift",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline scale shift";

    let arguments = (ins
        Optional<AnyMemRef>:$scale_table,
        OptionalAttr<IntAttr>:$shift_static
    );

    let assemblyFormat = [{
        ($scale_table^ `:` type($scale_table))?
        (`shift_static` `(` $shift_static^ `)`)?
        attr-dict
    }];

    let verifier = [{
        return vpux::VPUIPDPU::verifyOp(*this);
    }];
}

//
// PPEIntPreluShiftOp
//

def VPUIPDPU_PPEIntPreluShiftOp :
        VPUIPDPU_Op<"PPEIntPreluShift",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline prelu shift";

    let arguments = (ins
        IntAttr:$prelu_shift_static
    );

    let assemblyFormat = [{
        `prelu_shift_static` `(` $prelu_shift_static `)`
        attr-dict
    }];
}

//
// PPEIntRoundOp
//

def VPUIPDPU_PPEIntRoundOp :
        VPUIPDPU_Op<"PPEIntRound",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline rounding mode";

    let arguments = (ins
        VPUIPDPU_PPEIntRoundMode:$round_mode
    );

    let assemblyFormat = [{
        `round_mode` `(` $round_mode `)`
        attr-dict
    }];
}

//
// PPEIntZeroPointOffsetOp
//

def VPUIPDPU_PPEIntZeroPointOffsetOp :
        VPUIPDPU_Op<"PPEIntZeroPointOffset",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline zero point";

    let arguments = (ins
        IntAttr:$zero_point_static
    );

    let assemblyFormat = [{
        `zero_point_static` `(` $zero_point_static `)`
        attr-dict
    }];
}

//
// PPEIntClampOp
//

def VPUIPDPU_PPEIntClampOp :
        VPUIPDPU_Op<"PPEIntClamp",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline clamping";

    let description = [{
        In case PPE output is FP, only clamp_high is used by HW.
        Therefore the clamp_low param is optional, to be used only on the integer output data path.
    }];

    let arguments = (ins
        OptionalAttr<IntAttr>:$clamp_low,
        IntAttr:$clamp_high
    );

    let assemblyFormat = [{
        (`clamp_low` `(` $clamp_low^ `)`)?
        `clamp_high` `(` $clamp_high `)`
        attr-dict
    }];
}

//
// PPEIntConvertOp
//

def VPUIPDPU_PPEIntConvertOp :
        VPUIPDPU_Op<"PPEIntConvert",
            [
                HasParent<"vpux::VPUIPDPU::PPECfgOp">
            ]
        > {
    let summary = "PPE integer pipeline conversion";

    let arguments = (ins
        VPUIPDPU_PPEIntConvertMode:$convert_mode
    );

    let assemblyFormat = [{
        `convert_mode` `(` $convert_mode `)`
        attr-dict
    }];
}

//
// ODUCfg
//

def VPUIPDPU_ODUCfgOp :
        VPUIPDPU_Op<"ODUCfg",
            [
                HasParent<"vpux::VPUIPDPU::DPUInvariant">,
                NoTerminator
            ]
        > {
    let summary = "ODU configuration";

    let regions = (region
        SizedRegion<1>:$region
    );

    let assemblyFormat = [{
        attr-dict
        $region
    }];

    let verifier = [{
        return vpux::VPUIPDPU::verifyOp(*this);
    }];
}

//
// ODUOutTensorSizeOp
//

def VPUIPDPU_ODUOutTensorSizeOp :
        VPUIPDPU_Op<"ODUOutTensorSize",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU output tensor dimensions";

    let arguments = (ins
        IntAttr:$dim_x,
        IntAttr:$dim_y,
        IntAttr:$dim_z
    );

    let assemblyFormat = [{
        `dim_x` `(` $dim_x `)`
        `dim_y` `(` $dim_y `)`
        `dim_z` `(` $dim_z `)`
        attr-dict
    }];
}

//
// ODUDataReuseOp
//

def VPUIPDPU_ODUDataReuseOp :
        VPUIPDPU_Op<"ODUDataReuse",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU data type";

    let arguments = (ins

        VPUIPDPU_ODUActivationReuseMode:$activation_reuse
    );

    let assemblyFormat = [{
        `activation_reuse` `(` $activation_reuse `)`
        attr-dict
    }];
}

//
// ODUPermuteDataOp
//

def VPUIPDPU_ODUPermuteDataOp :
        VPUIPDPU_Op<"ODUPermuteData",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU data permutation mode";

    let arguments = (ins
        VPUIPDPU_ODUPermuteDataMode:$permute_mode
    );

    let assemblyFormat = [{
        `permute_mode` `(` $permute_mode `)`
        attr-dict
    }];
}

//
// ODUSparsityOp
//

def VPUIPDPU_ODUSparsityOp :
        VPUIPDPU_Op<"ODUSparsity",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU data sparsity config.";

    let description = [{
The default behaviour regarding sparsity (in case this op is not specified in the IR):
write_sp = 0, sp_out_en = 0 → dense activation written to ac_base; no sparsity output

Valid combinations of params:
| sparsity_map      | compression_enabled | sparse_value      | Behaviour                                                                                                    |
|:------------------|:--------------------|:------------------|:-------------------------------------------------------------------------------------------------------------|
| param present     | param not present   | param optional    | write_sp = 1, sp_out_en = 1 → sparse activation written to ac_base; sparsity map written to sp_base          |
| param present     | eq FALSE            | param not present | write_sp = 1, sp_out_en = 0 → dense activation written to ac_base; "dummy" sparsity map written to sp_base |
| param present     | eq TRUE             | param optional    | write_sp = 1, sp_out_en = 1 → sparse activation written to ac_base; sparsity map written to sp_base          |
| param not present | eq TRUE             | param optional    | write_sp = 0, sp_out_en = 1 → sparse activation written to ac_base; no sparsity output                       |

sparse_value is used by the ODU to determine sparsity
        }];

    let arguments = (ins
        Optional<AnyMemRef>:$sparsity_map,
        OptionalAttr<BoolAttr>:$compression_enabled,
        OptionalAttr<IntAttr>:$sparse_value
    );

    let assemblyFormat = [{
        ($sparsity_map^ `:` type($sparsity_map))?
        (`compression_enabled` `(` $compression_enabled^ `)`)?
        (`sparse_value` `(` $sparse_value^ `)`)?
        attr-dict
    }];

    let verifier = [{
        return vpux::VPUIPDPU::verifyOp(*this);
    }];
}

//
// ODUSwizzleDataOp
//

def VPUIPDPU_ODUSwizzleDataOp :
        VPUIPDPU_Op<"ODUSwizzleData",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU swizzle write out data config.";

    let arguments = (ins
        VPUIPDPU_ODUSwizzleKey:$swizzle_key
    );

    let assemblyFormat = [{
        `swizzle_key` `(` $swizzle_key `)`
        attr-dict
    }];
}

//
// ODUOutActivationsOp
//

def VPUIPDPU_ODUOutActivationsOp :
        VPUIPDPU_Op<"ODUOutActivations",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU activations output config.";

    let description = [{
Data type specification:
| attr name  | value range                                | vpu arch |
|:-----------|:-------------------------------------------|:---------|
| data_type  | fp16, u8f, g8, i8, i32, i4, i2, log, bin   | VPU2.7   |
If data_type is not specified then ODU output data type should be inferred from %out_activations.
        }];

    let arguments = (ins
        AnyMemRef:$out_activations,
        OptionalAttr<VPUIPDPU_ODUDataTypes>:$data_type
    );

    let assemblyFormat = [{
        `out_activations` `(` $out_activations `:` type($out_activations) `)`
        (`data_type` `(` $data_type^ `)`)?
        attr-dict
    }];

    let verifier = [{
        return vpux::VPUIPDPU::verifyOp(*this);
    }];
}

//
// ODUMemoryModeOp
//

def VPUIPDPU_ODUMemoryModeOp :
        VPUIPDPU_Op<"ODUMemoryMode",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">
            ]
        > {
    let summary = "ODU memory mode config.";

    let arguments = (ins
        VPUIPDPU_ODUMemoryMode:$mem_mode
    );

    let assemblyFormat = [{
        `mem_mode` `(` $mem_mode `)`
        attr-dict
    }];
}

//
// ODUCastOp
//

def VPUIPDPU_ODUCastOp :
        VPUIPDPU_Op<"ODUCast",
            [
                HasParent<"vpux::VPUIPDPU::ODUCfgOp">,
                VPUIPDPU_ArchKindVPUX37XX
            ]
        > {
    let summary = "ODU cast config.";

    let arguments = (ins
        AnyMemRef:$cast_output
    );

    let assemblyFormat = [{
        `cast_output` `(` $cast_output `:` type($cast_output) `)`
        attr-dict
    }];
}

//
// ODUOutSubtensorOp
//

def VPUIPDPU_ODUOutSubtensorOp :
        VPUIPDPU_Op<"ODUOutSubtensor",
            [
                HasParent<"vpux::VPUIPDPU::DPUVariant">
            ]
        > {
    let summary = "ODU output subtensor descriptor - begin and end coordinates";

    let arguments = (ins
        IntAttr:$begin_coord_x,
        IntAttr:$begin_coord_y,
        IntAttr:$begin_coord_z,
        IntAttr:$end_coord_x,
        IntAttr:$end_coord_y,
        IntAttr:$end_coord_z
    );

    let assemblyFormat = [{
        `begin_coord_x` `(` $begin_coord_x `)`
        `begin_coord_y` `(` $begin_coord_y `)`
        `begin_coord_z` `(` $begin_coord_z `)`
        `end_coord_x` `(` $end_coord_x `)`
        `end_coord_y` `(` $end_coord_y `)`
        `end_coord_z` `(` $end_coord_z `)`
        attr-dict
    }];
}

//
// MPEDenormalOperandsFTZ
//

def VPUIPDPU_MPEDenormalOperandsFTZOp :
        VPUIPDPU_Op<"MPEDenormalOperandsFTZ",
            [
                HasParent<"vpux::VPUIPDPU::MPECfgOp">
            ]
        > {
    let summary = "MPE force denormal operands to zero op";

    let assemblyFormat = [{
        attr-dict
    }];
}

//
// MPEActivationBias
//

def VPUIPDPU_MPEActivationBiasOp :
        VPUIPDPU_Op<"MPEActivationBias",
            [
                HasParent<"vpux::VPUIPDPU::MPECfgOp">
            ]
        > {
    let summary = "MPE set activation bias for U8 or I8 input op";

    let description = [{
        Used for asymetric integer quantization.
        The bias value (zero point) is treated as U8 if input is U8 or I8 if input is I8.
        Thus the op will restrict the bias value as such.
        For all other data types it is disregarded.
        }];
    let arguments = (ins
        IntAttr:$act_bias
    );

    let assemblyFormat = [{
        `act_bias` `(` $act_bias `)`
        attr-dict
    }];

    // TODO: E#59642 Add  verifyer type constraint to U8/I8 once IDU ops are ready

}

//
// MPEWeightsBias
//

def VPUIPDPU_MPEWeightsBiasOp :
        VPUIPDPU_Op<"MPEWeightsBias",
            [
                HasParent<"vpux::VPUIPDPU::MPECfgOp">
            ]
        > {
    let summary = "MPE set weights bias for U8 or I8 input op";

    let description = [{
        Used for asymetric integer quantization.
        The bias value (zero point) is treated as U8 if input is U8 or I8 if input is I8.
        Thus the op will restrict the bias value as such.
        For all other data types it is disregarded.
        }];

    let arguments = (ins
        IntAttr:$weights_bias
    );

    let assemblyFormat = [{
        `weights_bias` `(` $weights_bias `)`
        attr-dict
    }];
    // TODO: E#59642 Add  verifyer type constraint to U8/I8 once IDU ops are ready

}

#endif

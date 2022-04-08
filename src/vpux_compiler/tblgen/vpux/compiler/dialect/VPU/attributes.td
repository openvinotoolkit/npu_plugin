//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPU_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPU_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/VPU/dialect.td"

class VPU_Attr<string name, list<Trait> traits = []> : AttrDef<VPU_Dialect, name, traits> {
    let mnemonic = name;
}

//
// ArchKind
//

def VPU_ArchKind :
        StrEnumAttr<
            "ArchKind",
            "Represents VPU architecture generation",
            [
                StrEnumAttrCase<"UNKNOWN">,
                StrEnumAttrCase<"VPUX30XX">,
                StrEnumAttrCase<"VPUX311X">,
                StrEnumAttrCase<"VPUX37XX">,
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// MemoryKind
//

def VPU_MemoryKind :
        StrEnumAttr<
            "MemoryKind",
            "Represents the actual hardware memory hierarchy",
            [
                StrEnumAttrCase<"DDR">,
                StrEnumAttrCase<"CSRAM">,
                StrEnumAttrCase<"CMX_UPA">,
                StrEnumAttrCase<"CMX_NN">,
                StrEnumAttrCase<"Register">,
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// ExecutorKind
//

def VPU_ExecutorKind :
        StrEnumAttr<
            "ExecutorKind",
            "Representd hardware executror resources",
            [
                StrEnumAttrCase<"DMA_NN">,
                StrEnumAttrCase<"NCE">,
                StrEnumAttrCase<"DPU">,
                StrEnumAttrCase<"SHAVE_UPA">,
                StrEnumAttrCase<"SHAVE_NN">,
                StrEnumAttrCase<"SHAVE_ACT">,
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// CompilationMode
//

def VPU_CompilationMode  :
        StrEnumAttr<
            "CompilationMode",
            "Compilation Mode",
            [
                StrEnumAttrCase<"ReferenceSW">,
                StrEnumAttrCase<"ReferenceHW">,
                StrEnumAttrCase<"DefaultHW">
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// SparsitySupport
//

def VPU_SparsitySupport :
        BitEnumAttr<
            "SparsitySupport",
            "Sparsity support of an operation",
            [
                BitEnumAttrCase<"NONE", 0x0>,
                BitEnumAttrCase<"SPARSE_INPUTS", 0x1>,
                BitEnumAttrCase<"SPARSE_OUTPUTS", 0x2>,
                BitEnumAttrCase<"SPARSE_WEIGHTS", 0x4>
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// ActivationSparsityProfile
//

def VPU_ActivationSparsityProfile :
        StrEnumAttr<
            "ActivationSparsityProfile",
            "Represents desired activation sparsity profile",
            [
                StrEnumAttrCase<"S0">,      // Only for ops where runtime sparsity is possible
                StrEnumAttrCase<"S1">,      // As much as possible
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 0;
}

//
// WeightsSparsityHeuristic
//

def VPU_WeightsSparsityHeuristic :
        StrEnumAttr<
            "WeightsSparsityHeuristic",
            "Selects the weights sparsity heuristic which compares the sparse values ration to a threshold",
            [
                StrEnumAttrCase<"RATIO">,    // Fixed threshold based on the element type
                StrEnumAttrCase<"CMX">,      // Threshold is decided based on the CMX usage of the weights
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 0;
}

//
// EltwiseType
//

def VPU_EltwiseType :
        StrEnumAttr<
            "EltwiseType",
            "Type of Eltwise operation",
            [
                StrEnumAttrCase<"ADD">,
                StrEnumAttrCase<"SUBTRACT">,
                StrEnumAttrCase<"MULTIPLY">,
                StrEnumAttrCase<"DIVIDE">,
                StrEnumAttrCase<"SQUARED_DIFF">,
                StrEnumAttrCase<"POWER">,
                StrEnumAttrCase<"FLOOR_MOD">,
                StrEnumAttrCase<"MIN">,
                StrEnumAttrCase<"MAX">,
                StrEnumAttrCase<"AND">,
                StrEnumAttrCase<"EQUAL">,
                StrEnumAttrCase<"LESS">,
                StrEnumAttrCase<"LESS_EQUAL">,
                StrEnumAttrCase<"NOT_EQUAL">,
                StrEnumAttrCase<"GREATER">,
                StrEnumAttrCase<"GREATER_EQUAL">,
                StrEnumAttrCase<"LOGICAL_NOT">,
                StrEnumAttrCase<"LOGICAL_OR">,
                StrEnumAttrCase<"LOGICAL_XOR">,
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// PaddingAttr
//

def VPU_PaddingAttr :
        StructAttr<
            "PaddingAttr",
            VPU_Dialect,
            [
                StructFieldAttr<"left",   IntAttr>,
                StructFieldAttr<"right",  IntAttr>,
                StructFieldAttr<"top",    IntAttr>,
                StructFieldAttr<"bottom", IntAttr>,
            ]
        >;

//
// MPEMode
//

def VPU_MPEMode :
        StrEnumAttr<
            "MPEMode",
            "MPE Mode",
            [
                StrEnumAttrCase<"VECTOR">,
                StrEnumAttrCase<"MATRIX">,
                StrEnumAttrCase<"VECTOR_FP16">,
                StrEnumAttrCase<"CUBOID_16x16">,
                StrEnumAttrCase<"CUBOID_8x16">,
                StrEnumAttrCase<"CUBOID_4x16">,
                StrEnumAttrCase<"NOP">
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// PPEMode
//

def VPU_PPEMode :
        StrEnumAttr<
            "PPEMode",
            "Post Processing Element Type",
            [
                // Low-level instructions
                StrEnumAttrCase<"STORE">,
                StrEnumAttrCase<"LOAD">,
                StrEnumAttrCase<"CLEAR">,
                StrEnumAttrCase<"NOOP">,
                StrEnumAttrCase<"HALT">,

                // Element-Wise Operations
                StrEnumAttrCase<"ADD">,
                StrEnumAttrCase<"SUB">,
                StrEnumAttrCase<"MULT">,
                StrEnumAttrCase<"MAXIMUM">,
                StrEnumAttrCase<"MINIMUM">,
                StrEnumAttrCase<"AND">,
                StrEnumAttrCase<"OR">,
                StrEnumAttrCase<"XOR">,

                // Activations
                StrEnumAttrCase<"LRELU">,
                StrEnumAttrCase<"LRELUX">,
                StrEnumAttrCase<"LPRELU">,
                StrEnumAttrCase<"CEIL">,
                StrEnumAttrCase<"FLOOR">,
                StrEnumAttrCase<"POW">,
                StrEnumAttrCase<"EXP">,
                StrEnumAttrCase<"SIGMOID">,
                StrEnumAttrCase<"TANH">,
                StrEnumAttrCase<"SQRT">,
                StrEnumAttrCase<"RSQRT">,
                StrEnumAttrCase<"FLEXARB">,
                StrEnumAttrCase<"NOT">,
                StrEnumAttrCase<"ABS">,
                StrEnumAttrCase<"NEG">
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// PPETaskAttr
//

def VPU_PPETaskAttr :
        StructAttr<
            "PPETaskAttr",
            VPU_Dialect,
            [
                StructFieldAttr<"mode",             VPU_PPEMode>,
                StructFieldAttr<"clamp_low",        OptionalAttr<IntAttr>>,
                StructFieldAttr<"clamp_high",       OptionalAttr<IntAttr>>,
                StructFieldAttr<"lrelu_mult",       OptionalAttr<IntAttr>>,
                StructFieldAttr<"lrelu_shift",      OptionalAttr<IntAttr>>,
                StructFieldAttr<"quant_scale",      OptionalAttr<F64ArrayAttr>>,
                StructFieldAttr<"quant_mult",       OptionalAttr<I64ArrayAttr>>,
                StructFieldAttr<"quant_shift",      OptionalAttr<I64ArrayAttr>>,
                StructFieldAttr<"quant_post_shift", OptionalAttr<IntAttr>>,
                StructFieldAttr<"in1_quant_mult",       OptionalAttr<I64ArrayAttr>>,
                StructFieldAttr<"in2_quant_mult",       OptionalAttr<I64ArrayAttr>>,
                StructFieldAttr<"fp_prelu_alpha",   OptionalAttr<F64Attr>>,
            ]
        >;

//
// MultiClusterStrategy
//
def VPU_MultiClusterStrategy :
        StrEnumAttr<
            "MultiClusterStrategy",
            "MultiCluster Strategy",
            [
                StrEnumAttrCase<"SplitOverHeight">,
                StrEnumAttrCase<"SplitOverKernel">,
                StrEnumAttrCase<"SplitOverWidth">,
                StrEnumAttrCase<"Clustering">,
                StrEnumAttrCase<"SplitOverHeightOverlapped">,
                StrEnumAttrCase<"HKSwitch">
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// DistributionMode
//

def VPU_DistributionMode :
        BitEnumAttr<
            "DistributionMode",
            "Tensor distribution mode",
            [
                BitEnumAttrCase<"NONE",         0x0>,
                BitEnumAttrCase<"OVERLAPPED",   0x1>,
                BitEnumAttrCase<"DUPLICATED",   0x2>,
                BitEnumAttrCase<"SEGMENTED",    0x4>,
                BitEnumAttrCase<"MULTICASTED",  0x8>
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// DistributedTensorAttr
//

def VPU_DistributedTensorAttr :
        StructAttr<
            "DistributedTensorAttr",
            VPU_Dialect,
            [
                StructFieldAttr<"mode", VPU_DistributionMode>,
                StructFieldAttr<"num_tiles", OptionalAttr<I64ArrayAttr>>,
                StructFieldAttr<"kernel", OptionalAttr<I64ArrayAttr>>,
                StructFieldAttr<"pads", OptionalAttr<VPU_PaddingAttr>>,
                StructFieldAttr<"strides", OptionalAttr<I64ArrayAttr>>,
                StructFieldAttr<"num_clusters", IntAttr>,
                StructFieldAttr<"alignment", OptionalAttr<I64ArrayAttr>>
            ]
        >;

//
// CompressionSchemeAttr
//

def VPU_CompressionSchemeAttr : VPU_Attr<"CompressionScheme"> {
    let description = [{
        Represents the compression as the number of elements along a specified axis.

        For example, a two-dimensional type with the shape 4x30 might be compressed
        along axis 0 into with the number of elements [12, 15, 30, 3].

        In case the compression is over the entire data (instead of a specified axis),
        the `axis` attribute can be set to null with the `numElems` as a splat value.

        The `alignment` attribute can be used to represent a required alignment for
        each set of elements on the given axis. For example, in case the compression
        for weights sparsity is represented by this attribute, the compression will
        be over the output channel axis and each weight set (i.e. ICxKYxKX - set of
        values for each output channel) has to be aligned to 16 bytes.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$axis,
        "mlir::ElementsAttr":$numElems,
        "mlir::IntegerAttr":$alignment
    );

    let extraClassDeclaration = [{
        int64_t getTotalNumElems() const;
        int64_t getNumElemsInRange(int64_t startIdx, int64_t size) const;
        Byte getAllocSize(mlir::Type elemType) const;
    }];

    let assemblyFormat = "`<` struct(params) `>`";
}

#endif

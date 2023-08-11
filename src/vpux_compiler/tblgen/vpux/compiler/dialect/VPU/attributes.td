//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPU_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/VPU/dialect.td"
include "vpux/compiler/dialect/VPU/attr_interfaces.td"


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
                StrEnumAttrCase<"DefaultHW">,
                StrEnumAttrCase<"ShaveCodeGen">
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// SparsitySupport
//

def VPU_SparsitySupport :
        I32BitEnumAttr<
            "SparsitySupport",
            "Sparsity support of an operation",
            [
                I32BitEnumAttrCase<"NONE", 0x0>,
                I32BitEnumAttrCase<"SPARSE_INPUTS", 0x1>,
                I32BitEnumAttrCase<"SPARSE_OUTPUTS", 0x2>,
                I32BitEnumAttrCase<"SPARSE_WEIGHTS", 0x4>
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
// NCE Interpolate
//

def VPU_NCEInterpolateMode :
        StrEnumAttr<
            "NCEInterpolateMode",
            "Specifies type of interpolation",
            [
                StrEnumAttrCase<"NEAREST">,
                StrEnumAttrCase<"BILINEAR">
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// DistributionMode
//

def VPU_DistributionMode :
        I64BitEnumAttr<
            "DistributionMode",
            "Tensor distribution mode",
            [
                I64BitEnumAttrCase<"NONE",         0x0>,
                I64BitEnumAttrCase<"OVERLAPPED",   0x1>,
                I64BitEnumAttrCase<"DUPLICATED",   0x2>,
                I64BitEnumAttrCase<"SEGMENTED",    0x4>,
                I64BitEnumAttrCase<"MULTICASTED",  0x8>
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// DistributedTensorAttr
//

/////////////////////////////
//
// mode - hint for how data and compute are distributed across clusters
//      * SEGMENTED - data and compute is split in a simple manner across clusters
//      not taking into account the overlap lines. This is possible for
//      input tensors in VPUX3XXX since HW can read the overlap lines from
//      neighboring clusters.
//      * DUPLICATED - both data and compute are duplicated on the specified clusters.
//      * SEGMENTED | DUPLICATED - such a combination means compute is segmented across clusters
//      but the data will be duplicated with the feature of broadcast.
//      * SEGMENTED | MULTICASTED - similar to SEGMENTED | DUPLICATED but used for cases when we
//      broadcast the data and segment across the channel dimension. In future this will be removed
//      in favor of only SEGMENTED | DUPLICATED
// num_tiles - shape for tiling data and compute across clusters
//      * size of num_tiles is equal to the data shape size
//      * amount of tiling per one axis is usually equal to num_clusters
//      unless tiling is done on multiple axes
// kernel/pads/strides - parameters used to compute overlap lines in the case of
//      OVERLAP mode, logic works by infering a balanced split of the result buffer
//      by taking into account the current buffer as input followed by a backinfer
//      of the input shapes per cluster with the mentioned overlap params, such that
//      the will be produced accordingly
// num_clusters - amount of clusters over which the data and compute is segmented
//      * this is not necessarily always equal to the full amount of clusters available
//      during a compilation, for end-end performance reasons
// alignment - shape describing how the per cluster tiled shapes should be aligned
//      thus the per cluster segmentation is done with this alignment in mind
// uniform_distributed_segments - boolean controling the approach of how data and compute are split
//      across clusters.
//      * for example splitting 10 compute lines across 4 clusters is done like:
//          * [3, 3, 2, 2] when uniform_distributed_segments = true
//          * [3, 3, 3, 1] when uniform_distributed_segments = false
//      * in absence of explicit reasons or limitations, uniform_distributed_segments = true
//      is preferred since it generates a more uniformly distribution of compute and data
// compute_shapes - array of shapes per cluster used with OVERLAP mode.
//      * represents exact shapes found in each cluster
//      * assumes both compute (result of a compute op) & memory (what is actually in memory)
//        shapes are equal
//      * mutually exclusive with kernel/pad/strides
// compute_offsets - array of offsets per cluster used with OVERLAP mode.
//      * represents exact offsets found in each cluster
//      * assumes both compute (result of a compute op) & memory (what is actually in memory)
//        offsets are equal
//      * mutually exclusive with kernel/pad/strides
// equal_memory_and_compute_view - used with OVERLAP mode
//      * indicates compute view should be obtained by applying memory view calculations
//      * necessary when having output OVERLAPPED
//
/////////////////////////////

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
                StructFieldAttr<"alignment", OptionalAttr<I64ArrayAttr>>,
                StructFieldAttr<"uniform_distributed_segments", OptionalAttr<UnitAttr>>,
                StructFieldAttr<"compute_shapes", OptionalAttr<I64ArrayOfArraysAttr>>,
                StructFieldAttr<"compute_offsets", OptionalAttr<I64ArrayOfArraysAttr>>,
                StructFieldAttr<"equal_memory_and_compute_view", OptionalAttr<UnitAttr>>
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

//
// SEAttr
//

class VPU_SEAttrBase<Pred condition, string summary> : Attr<condition, summary> {
    let storageType = [{ vpux::VPU::SEAttr }];
    let returnType = [{ vpux::VPU::SEAttr }];
    let convertFromStorage = "$_self";
}

def VPU_SEAttr : VPU_SEAttrBase<CPred<"$_self.isa<vpux::VPU::SEAttr>()">,
                                      "Storage Element attribute"> {
        string cppType = "vpux::VPU::SEAttr";
    }


//
// NCEInterpolateCoordMode
//

def VPU_NCEInterpolateCoordMode :
        StrEnumAttr<
            "NCEInterpolateCoordMode",
            "coordinate_transformation_mode specifies how to transform the coordinate.",
            [
                StrEnumAttrCase<"ASYMMETRIC">
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// NCEInterpolateNearestMode
//

def VPU_NCEInterpolateNearestMode :
        StrEnumAttr<
            "NCEInterpolateNearestMode",
            "specifies round mode when mode == nearest",
            [
                StrEnumAttrCase<"ROUND_PREFER_FLOOR">,
                StrEnumAttrCase<"ROUND_PREFER_CEIL">,
                StrEnumAttrCase<"FLOOR">,
                StrEnumAttrCase<"CEIL">,
                StrEnumAttrCase<"SIMPLE">,
            ]
        > {
    let cppNamespace = "vpux::VPU";
    let genSpecializedAttr = 1;
}

//
// SEInterpolateAttr
//

def VPU_SEInterpolateAttr : VPU_Attr<"SEInterpolate", [
        DeclareAttrInterfaceMethods<VPU_SEAttrInterface>
    ] > {
    let description = [{
        This attribute contains parameters for HW interpolate which is implemented by means of storage element table.
        It represents how the storage element table is generated to pick elements from the initial data.
        It is intended to use with input sparse type for NCE operations and the storage element table operation.

        Scaling function is chosen based on interpolate mode, nearest mode and coordination transformation mode parameters.

        Scales used to calculate an output shape together with scaling function deduced from the paramaters mentioned above.
        The offsets and sizes attributes refer to the part of the input data that is extracted after interpolation.
    }];
    let parameters = (ins
        "vpux::VPU::NCEInterpolateModeAttr":$mode,
        "vpux::VPU::NCEInterpolateNearestModeAttr":$nearest_mode,
        "vpux::VPU::NCEInterpolateCoordModeAttr":$coordinate_transformation_mode,
        "mlir::ArrayAttr":$scale,
        "mlir::ArrayAttr":$offsets,
        "mlir::ArrayAttr":$sizes
    );

    let assemblyFormat = "`<` struct(params) `>`";
    let genVerifyDecl = 1;
}

#endif

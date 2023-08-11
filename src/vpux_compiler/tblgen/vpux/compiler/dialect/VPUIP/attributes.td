//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPUIP_ATTRIBUTES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"

class VPUIP_Attr<string name, list<Trait> traits = []> : AttrDef<VPUIP_Dialect, name, traits> {
    let mnemonic = name;
}

//
// PoolLayerType
//

def VPUIP_PoolLayerType :
        StrEnumAttr<
            "PoolLayerType",
            "Type of Pooling layer",
            [
                StrEnumAttrCase<"MAX">,
                StrEnumAttrCase<"AVG">,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

//
// ReduceLayerType
//

def VPUIP_ReduceLayerType :
        StrEnumAttr<
            "ReduceLayerType",
            "Type of Reduce layer",
            [
                StrEnumAttrCase<"MAX">,
                StrEnumAttrCase<"MEAN">,
                StrEnumAttrCase<"LOGICAL_OR">,
                StrEnumAttrCase<"LOGICAL_AND">,
                StrEnumAttrCase<"PROD">,
                StrEnumAttrCase<"SUM">,
                StrEnumAttrCase<"MIN">,
                StrEnumAttrCase<"L1">,
                StrEnumAttrCase<"L2">,
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

//
// NCETaskType
//

def VPUIP_NCETaskType :
        StrEnumAttr<
            "NCETaskType",
            "NCE task type",
            [
                StrEnumAttrCase<"CONV">,
                StrEnumAttrCase<"DWCONV">,
                StrEnumAttrCase<"MAXPOOL">,
                StrEnumAttrCase<"AVEPOOL">,
                StrEnumAttrCase<"FCL">,
                StrEnumAttrCase<"ELTWISE">,
                StrEnumAttrCase<"IDENTITY">,
                StrEnumAttrCase<"CMCONV">
            ]
        > {
    let cppNamespace = "vpux::VPUIP";
    let genSpecializedAttr = 1;
}

//
// CompressionSchemeAttr
//

def VPUIP_CompressionSchemeAttr : VPUIP_Attr<"CompressionScheme"> {
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

def VPUIP_SwizzlingSchemeAttr : VPUIP_Attr<"SwizzlingScheme"> {
    let description = [{
        This contains information about all the swizzling related requirements for the buffer
        
        The 'key' is the swizzling key to be used, these keys refers to 5 swizzling patterns that determines how much striping is applied
        Higher swizzling shows higher performance but needs larger alignment/padding size for each allocation depending on the architecture
        The 'sizeAlignment' is the requirement for the buffer size dictated by the architecture
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$key,
        "mlir::IntegerAttr":$sizeAlignment
    );
    let assemblyFormat = "`<` struct(params) `>`";
}

//
// MemRefAttr
//

def VPUIP_MemRefAttr :
        StructAttr<
            "MemRefAttr",
            VPUIP_Dialect,
            [
                StructFieldAttr<"order", AffineMapAttr>,
                StructFieldAttr<"strides", OptionalAttr<I64ArrayAttr>>,
                StructFieldAttr<"swizzlingScheme", OptionalAttr<VPUIP_SwizzlingSchemeAttr>>,
                StructFieldAttr<"compressionScheme", OptionalAttr<VPUIP_CompressionSchemeAttr>>,
                StructFieldAttr<"allocSize", OptionalAttr<IntAttr>>
            ]
        >;

//
// DmaDescriptorAttr
//

def VPUIP_DmaDescriptorAttr :
        StructAttr<
            "DmaDescriptorAttr",
            VPUIP_Dialect,
            [
                StructFieldAttr<"numPlanes", IntAttr>,
                StructFieldAttr<"len", IntAttr>,
                StructFieldAttr<"srcWidth", IntAttr>,
                StructFieldAttr<"srcStride", IntAttr>,
                StructFieldAttr<"srcPlaneStride", IntAttr>,
                StructFieldAttr<"dstWidth", IntAttr>,
                StructFieldAttr<"dstStride", IntAttr>,
                StructFieldAttr<"dstPlaneStride", IntAttr>
            ]
        >;

//
// DmaChannelType
//

def VPUIP_DmaChannelType :
        I64EnumAttr<
            "DmaChannelType",
            "DMA channel type",
            [
                I64EnumAttrCase<"DDR", 0>,
                I64EnumAttrCase<"CMX", 1>
            ]
        > {
    let cppNamespace = "vpux::VPUIP";

    let description = [{
      In VPU different DMA channel receives data movement job from link agent.
      Different channel is used based on transaction source:
       - DDR channel - DMA SRC is DDR
       - CMX channel - DMA SRC is CMX or HW register"
    }];
}

#endif

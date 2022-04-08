//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_TYPES
#define VPUX_COMPILER_DIALECT_VPUIP_TYPES

include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPU/type_interfaces.td"
include "vpux/compiler/core/type_interfaces.td"

include "mlir/IR/SubElementInterfaces.td"

// Base class for VPUIP dialect types.
class VPUIP_Type<string name, list<Trait> traits = []>
    : TypeDef<VPUIP_Dialect, name, traits> {
  let mnemonic = name;
}

//
// Buffer
//

def VPUIP_Buffer :
        VPUIP_Type<
            "Buffer",
            [
                DeclareTypeInterfaceMethods<NDTypeInterface>
            ]
        > {
    let summary = "VPUIP Buffer Type";

    let description = "This type represents a simple buffer similar to memref";

    let genAccessors = 0;

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "mlir::Type":$elementType,
        "mlir::MemRefLayoutAttrInterface":$layout,
        "vpux::IndexedSymbolAttr":$memSpace,
        "mlir::IntegerAttr":$swizzlingKey
    );

    let extraClassDeclaration = [{
        mlir::MemRefLayoutAttrInterface getLayout() const;
        mlir::IntegerAttr getSwizzlingKey() const;
    }];

    let builders = [
        TypeBuilderWithInferredContext<
            (ins
                "::mlir::MLIRContext* ":$context,
                "::llvm::ArrayRef<int64_t>":$shape,
                "mlir::Type":$elementType,
                "mlir::MemRefLayoutAttrInterface":$layout,
                "vpux::IndexedSymbolAttr":$memSpace
            ), [{
            return $_get(context, shape, elementType, layout, memSpace, nullptr);
        }]>
    ];

}

//
// DistributedBuffer
//

def VPUIP_DistributedBuffer :
        VPUIP_Type<
            "DistributedBuffer",
            [
                DeclareTypeInterfaceMethods<SubElementTypeInterface>,
                DeclareTypeInterfaceMethods<NDTypeInterface>,
                DeclareTypeInterfaceMethods<ClusterTypeInterface>
            ]
        > {
    let summary = "VPUIP buffer type to describe the buffer tiling";
    let description = [{ This type of buffer is used together with the ClusterTiling operation
                            to describe a tile operation between clusters }];
    let genAccessors = 0;

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "mlir::Type":$elementType,
        "mlir::MemRefLayoutAttrInterface":$layout,
        "vpux::IndexedSymbolAttr":$memSpace,
        "VPU::DistributedTensorAttr":$distribution,
        "VPUIP::CompressionSchemeAttr":$compressionScheme
    );

    let builders = [
        TypeBuilderWithInferredContext<
            (ins
                "mlir::MLIRContext*":$context,
                "llvm::ArrayRef<int64_t>":$shape,
                "mlir::Type":$elementType,
                "mlir::MemRefLayoutAttrInterface":$layout,
                "vpux::IndexedSymbolAttr":$memSpace,
                "VPU::DistributedTensorAttr":$distribution,
                CArg<"VPUIP::CompressionSchemeAttr", "{}">:$compressionScheme
            ), [{
            return $_get(context, shape, elementType, layout, memSpace, distribution, compressionScheme);
        }]>
    ];

    let extraClassDeclaration = [{
        mlir::MemRefType getCompactType() const;

        mlir::MemRefLayoutAttrInterface getLayout() const;
        VPU::DistributedTensorAttr getDistribution() const;
        VPUIP::CompressionSchemeAttr getCompressionScheme() const;
    }];

    let skipDefaultBuilders = 1;
    let genVerifyDecl = 1;
}

//
// SparseBuffer
//

def VPUIP_SparseBuffer :
        VPUIP_Type<
            "SparseBuffer",
            [
                DeclareTypeInterfaceMethods<NDTypeInterface>
            ]
        > {
    let summary = "VPUIP Sparse Buffer Type";

    let description = [{
        This type represents a sparse buffer as a group of data and metadata.
        The metadata is represented by the sparsity map and, in some instances,
        the storage element table.

        The data and metadata have to be of one of the following types:
        - mlir::MemRefType
        - VPU::DistributedBufferType

        The `isWeights` attribute is used to mark cases where a sparse buffer is
        consumed as weights by the IDU of the user operation. In such cases, the
        weights set size of the sparsity map (i.e. ICxKYxKW) needs to be aligned
        to 16 bytes for every output channel.

        The `compressionScheme` attribute is utilized for weights sparsity to
        identify the number of elements per output channel that are present in
        the data after removing the sparse values.
    }];

    let parameters = (ins
        "mlir::Type":$data,
        "mlir::Type":$sparsityMap,
        "mlir::Type":$storageElementTable,
        "mlir::UnitAttr":$isWeights,
        "VPUIP::CompressionSchemeAttr":$compressionScheme
    );

    let builders = [
        TypeBuilderWithInferredContext<
            (ins
                "mlir::Type":$data,
                CArg<"mlir::Type", "{}">:$sparsityMap,
                CArg<"mlir::Type", "{}">:$storageElementTable,
                CArg<"mlir::UnitAttr", "{}">:$isWeights,
                CArg<"VPUIP::CompressionSchemeAttr", "{}">:$compressionScheme
            ), [{
            return $_get(data.getContext(), data, sparsityMap, storageElementTable, isWeights, compressionScheme);
        }]>
    ];

    let skipDefaultBuilders = 1;
    let genVerifyDecl = 1;
}

#endif

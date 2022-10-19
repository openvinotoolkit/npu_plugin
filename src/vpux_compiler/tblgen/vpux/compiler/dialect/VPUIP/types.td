//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_TYPES
#define VPUX_COMPILER_DIALECT_VPUIP_TYPES

include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPU/type_interfaces.td"
include "vpux/compiler/core/type_interfaces.td"

include "mlir/IR/SubElementInterfaces.td"

// Base class for VPUIP dialect types.
class VPUIP_Type<string name, list<Trait> traits = []>
    : TypeDef<VPUIP_Dialect, name, traits> {
  let mnemonic = name;
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
        "VPU::DistributedTensorAttr":$distribution
    );

    let extraClassDeclaration = [{
        mlir::MemRefType getCompactType() const;

        mlir::MemRefLayoutAttrInterface getLayout() const;
        VPU::DistributedTensorAttr getDistribution() const;
    }];

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

    let description = "This object represents a set of memory references that compose sparse data";

    let parameters = (ins
        "mlir::MemRefType":$data,
        "mlir::MemRefType":$sparsityMap,
        "mlir::MemRefType":$storageElementTable
    );

    let builders = [
        TypeBuilderWithInferredContext<
            (ins
                "mlir::MemRefType":$data,
                CArg<"mlir::MemRefType", "{}">:$sparsityMap,
                CArg<"mlir::MemRefType", "{}">:$storageElementTable
            ), [{
            return $_get(data.getContext(), data, sparsityMap, storageElementTable);
        }]>
    ];

    let skipDefaultBuilders = 1;
}

#endif

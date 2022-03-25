//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_TYPES
#define VPUX_COMPILER_DIALECT_VPUIP_TYPES

include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/type_interfaces.td"
include "vpux/compiler/core/type_interfaces.td"
include "vpux/compiler/core/attributes.td"

include "mlir/IR/BuiltinDialect.td"
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

#endif

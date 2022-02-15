//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_TYPES
#define VPUX_COMPILER_DIALECT_VPUIP_TYPES

include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPU/attributes.td"
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
                DeclareTypeInterfaceMethods<NDTypeInterface>
            ]
        > {
    let summary = "VPUIP buffer type to describe the buffer tiling";
    let description = [{ This type of buffer is used together with the ClusterTiling operation
                            to describe a tile operation between clusters }];
    let genAccessors = 0;

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "mlir::Type":$elementType,
        "mlir::AffineMapAttr":$order,
        "vpux::IndexedSymbolAttr":$memSpace,
        "VPU::DistributedTensorAttr":$distribution
    );

    let extraClassDeclaration = [{
        mlir::MemRefType getCompactType() const;

        mlir::AffineMapAttr getOrder() const;
        VPU::DistributedTensorAttr getDistribution() const;
    }];
}

#endif

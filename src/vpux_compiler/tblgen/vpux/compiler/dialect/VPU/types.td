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

#ifndef VPUX_COMPILER_DIALECT_VPU_TYPES
#define VPUX_COMPILER_DIALECT_VPU_TYPES

include "vpux/compiler/dialect/VPU/dialect.td"
include "vpux/compiler/dialect/VPU/attributes.td"

include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/SubElementInterfaces.td"

// Base class for VPU dialect types.
class VPU_Type<string name, list<Trait> traits = []>
    : TypeDef<VPU_Dialect, name, traits> {
  let mnemonic = name;
}

//
// DistributedTensor
//

def VPU_DistributedTensor :
        VPU_Type<
            "DistributedTensor",
            [
                DeclareTypeInterfaceMethods<SubElementTypeInterface>
            ]
        > {
    let summary = "VPU tensor type to describe the tensor tiling";
    let description = [{ This type of tensor is used together with the ClusterTiling operation
                            to describe a tile operation between clusters }];
    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "mlir::Type":$elementType,
        "mlir::AffineMapAttr":$order,
        "vpux::IndexedSymbolAttr":$memSpace,
        "DistributedTensorAttr":$distribution
    );

    let extraClassDeclaration = [{
        mlir::RankedTensorType getCompactType() const;
    }];
}

#endif

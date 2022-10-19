//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPU_TYPES
#define VPUX_COMPILER_DIALECT_VPU_TYPES

include "vpux/compiler/core/type_interfaces.td"
include "vpux/compiler/dialect/VPU/dialect.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/type_interfaces.td"
include "vpux/compiler/core/type_interfaces.td"

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
                DeclareTypeInterfaceMethods<SubElementTypeInterface>,
                DeclareTypeInterfaceMethods<NDTypeInterface>,
                DeclareTypeInterfaceMethods<ClusterTypeInterface>
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

    let genAccessors = 0;

    let extraClassDeclaration = [{
        mlir::RankedTensorType getCompactType() const;

        mlir::AffineMapAttr getOrder() const;
        DistributedTensorAttr getDistribution() const;
    }];

    let genVerifyDecl = 1;
}

//
// SparseTensor
//

def VPU_SparseTensor :
        VPU_Type<
            "SparseTensor",
            [
                DeclareTypeInterfaceMethods<NDTypeInterface>
            ]
        > {
    let summary = "VPU SparseTensor Type";

    let description = "This object represents a sparse tensor as a group of data and metadata";

    let parameters = (ins
        "mlir::TensorType":$data,
        "mlir::TensorType":$sparsityMap,
        "mlir::TensorType":$storageElementTable
    );

    let builders = [
        TypeBuilderWithInferredContext<
            (ins
                "mlir::TensorType":$data,
                CArg<"mlir::TensorType", "{}">:$sparsityMap,
                CArg<"mlir::TensorType", "{}">:$storageElementTable
            ), [{
            return $_get(data.getContext(), data, sparsityMap, storageElementTable);
        }]>
    ];

    let skipDefaultBuilders = 1;
}

#endif

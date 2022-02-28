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
include "vpux/compiler/core/type_interfaces.td"

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
                DeclareTypeInterfaceMethods<SubElementTypeInterface>,
                DeclareTypeInterfaceMethods<NDTypeInterface>
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

        // @brief Retrive the array of compute shapes.
        // @warning An important thing to consider with regards to compute shapes,
        // is that modes like SEGMENTED and OVERLAPPED take precedence over
        // DUPLICATED and MULTICASTED.
        // In an example case of a "SEGMENTED | DUPLICATED" (needed for SplitOverK)
        // tensor with shape [1, 64, 4, 4], the compute shape in each cluster is
        // [1, 16, 4, 4], which is needed when tiling and generating workloads,
        // while the allocated shape is [1, 64, 4, 4] (because of duplicated)
        // information which is needed for scheduler and strategy manager,
        // in order to estimate memory
        mlir::SmallVector<vpux::Shape> getPerClusterComputeShapes() const;

        // @brief Get largest compact compute shape
        // @warning This function should not be used for memory size calculation,
        // because it does not retrieve the true allocate shape in cases
        // of broadcasting.
        vpux::Shape getLargestCompactShape() const;

        // @brief Get the compact compute shape for a specific cluster
        // @warning This function should not be used for memory size calculation,
        // because it does not retrieve the true allocate shape in cases
        // of broadcasting.
        vpux::Shape getCompactShape(int64_t tileInd) const;
    }];

    let genVerifyDecl = 1;
}

#endif

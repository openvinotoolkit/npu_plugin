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

#ifndef VPUX_COMPILER_DIALECT_VPU_TYPE_INTERFACES
#define VPUX_COMPILER_DIALECT_VPU_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

//
// ClusterTypeInterface
//

def ClusterTypeInterface : TypeInterface<"ClusterTypeInterface"> {
    let description = [{
        Interface for generating cluster-aware information for types.
    }];

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Retrive the array of compute shapes",
            "SmallVector<Shape>", "getPerClusterComputeShapes", (ins)
        >,

        InterfaceMethod<
            "Retrive the array of compute shape offsets with regards to the full buffer",
            "SmallVector<Shape>", "getPerClusterComputeShapeOffsets", (ins)
        >,

        InterfaceMethod<
            "Get largest compact compute shape",
            "Shape", "getLargestCompactShape", (ins)
        >,

        InterfaceMethod<
            "Get the compact compute shape for a specific cluster",
            "Shape", "getCompactShape", (ins "int64_t":$tileInd)
        >,

       InterfaceMethod<
            "Retrive the array of strided compute shapes",
            "SmallVector<std::pair<Shape, Strides>>", "getPerClusterStridedShapes", (ins)
        >,

        InterfaceMethod<
            "Get largest strided compute shape",
            "std::pair<Shape, Strides>", "getLargestStridedShape", (ins)
        >,

        InterfaceMethod<
            "Get the strided compute shape for a specific cluster",
            "std::pair<Shape, Strides>", "getStridedShape", (ins "int64_t":$tileInd)
        >,
    ];
}

#endif

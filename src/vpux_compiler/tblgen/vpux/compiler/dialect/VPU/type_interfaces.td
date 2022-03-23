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
        InterfaceMethod<[{
            @brief Retrieve the array of compute shapes
            @warning An important thing to consider with regards to compute shapes,
                     is that modes like SEGMENTED and OVERLAPPED take precedence over
                     DUPLICATED and MULTICASTED.
                     In an example case of a "SEGMENTED | DUPLICATED" (needed for SplitOverK)
                     tensor with shape [1, 64, 4, 4], the compute shape in each cluster is
                     [1, 16, 4, 4], which is needed when tiling and generating workloads,
                     while the allocated shape is [1, 64, 4, 4] (because of duplicated)
                     information which is needed for scheduler and strategy manager,
                     in order to estimate memory
            }],
            "SmallVector<Shape>", "getPerClusterComputeShapes", (ins)
        >,

        InterfaceMethod<[{
            @brief Retrieve the array of compute shape offsets with regards to the full buffer
            @warning An important thing to consider with regards to compute offsets,
                     is that modes like SEGMENTED and OVERLAPPED take precedence over
                     DUPLICATED and MULTICASTED.
             }],
            "SmallVector<Shape>", "getPerClusterComputeShapeOffsets", (ins)
        >,

        InterfaceMethod<[{
            @brief Get largest compact compute shape
            @warning This function should not be used for memory size calculation,
                     because it does not retrieve the true allocate shape in cases
                     of broadcasting.
            }],
            "Shape", "getLargestCompactShape", (ins)
        >,

        InterfaceMethod<[{
            @brief Get the compact compute shape for a specific cluster
            @warning This function should not be used for memory size calculation,
                     because it does not retrieve the true allocate shape in cases
                     of broadcasting.
            }],
            "Shape", "getCompactShape", (ins "int64_t":$tileInd)
        >,

        InterfaceMethod<[{
            @brief Retrieve the array of padding for each cluster
            @warning This function is needed for getting padding in OVERLAPPED mode
            }],
            "SmallVector<vpux::PadInfo>", "getPerClusterPadding", (ins)
        >,

       InterfaceMethod<[{
            @brief Retrieve the array of strided compute shapes
            @warning This function should not be used for memory size calculation,
                     because it does not retrieve the true allocate shape in cases
                     of broadcasting.
            }],
            "SmallVector<StridedShape>", "getPerClusterStridedShapes", (ins)
        >,

        InterfaceMethod<[{
            @brief Get largest strided compute shape
            @warning This function should not be used for memory size calculation,
                     because it does not retrieve the true allocate shape in cases
                     of broadcasting.
            }],
            "StridedShape", "getLargestStridedShape", (ins)
        >,

        InterfaceMethod<[{
            @brief Get the strided compute shape for a specific cluster
            @warning This function should not be used for memory size calculation,
                     because it does not retrieve the true allocate shape in cases
                     of broadcasting.
            }],
            "StridedShape", "getStridedShape", (ins "int64_t":$tileInd)
        >,
    ];
}

#endif

//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

namespace vpux {
namespace VPU {
struct OverlapDistributionParams {
    OverlapDistributionParams(mlir::ArrayAttr kernel, VPU::PaddingAttr pads, mlir::ArrayAttr stride,
                              mlir::UnitAttr equalComputeAndMemoryView = nullptr)
            : kernel(kernel), pads(pads), stride(stride), equalComputeAndMemoryView(equalComputeAndMemoryView){};

    OverlapDistributionParams(mlir::ArrayAttr memoryShapes, mlir::ArrayAttr memoryOffsets)
            : memoryShapes(memoryShapes), memoryOffsets(memoryOffsets){};

    mlir::ArrayAttr kernel = nullptr;
    VPU::PaddingAttr pads = nullptr;
    mlir::ArrayAttr stride = nullptr;
    mlir::UnitAttr equalComputeAndMemoryView = nullptr;
    mlir::ArrayAttr memoryShapes = nullptr;
    mlir::ArrayAttr memoryOffsets = nullptr;
};

// Looks over the ops in opSubgraph and selects those that implement NCEOpInterface and are SOH-compatible. From that
// subset, picks the Overlapped params with the largest kernel. Overlapped params are described by combinations of
// kernel, pads, strides.
OverlapDistributionParams getOverlappedDistributionParameters(mlir::MLIRContext* ctx,
                                                              ArrayRef<VPU::ClusteredOpInterface> opSubgraph,
                                                              int64_t kernelDistributionAxis,
                                                              mlir::UnitAttr equalComputeAndMemoryView = nullptr);

// For each cluster, computes the union of the memory views of the consumerSubgraph ops' inputs and the compute view of
// the producer op's output. The ops considered from consumerSubgraph must implement NCEOpInterface and be
// SOH-compatible. E.g:
//  _________________________________________________
// |               NCEOp0                            |
// | [0, 0, 0, 0] -> [1, 15, 13, 17] (compute view0) |
// | [0, 0, 14, 0] -> [1, 15, 27, 17] (compute view1)|
// |_________________________________________________|
//                        |
//  ________________________________________________
// |  [0, 0, 0, 0] -> [1, 15, 17, 17] (mem view0)   |
// |  [0, 0, 15, 0] -> [1, 15, 27, 17] (mem view1)  |
// |                 NCEOp1                         |
// |________________________________________________|
//
// Resulting OverlappedParams: cluster 0 = [0, 0, 0, 0] -> [1, 15, 17, 17], cluster 1 = [0, 0, 14, 0] -> [1, 15, 27, 17]
// OverlappedParams are described by explicit per cluster memory shapes and offsets.
OverlapDistributionParams getOverlappedDistributionParameters(mlir::MLIRContext* ctx,
                                                              VPU::ClusteredOpInterface producer,
                                                              ArrayRef<VPU::ClusteredOpInterface> consumerSubgraph,
                                                              const int64_t numClusters, ArrayRef<int64_t> numTiles,
                                                              mlir::UnitAttr uniformDistributedSegments);

// In case of input being presented with explicit overlap lines with DPU,
// we need to take into account all the siblings requirements
// when it comes to kernel, pad and stride.
//
// For the best handling, to provide the output which can service all siblings
// without spilling be required, we will need the support of precomputed shapes/offsets
// per cluster.
// This is because of the cases when different ops may have the maximum requirements
// on different clusters. In which case, there's no singular way with current distributed
// infrastructure to represent a mixed tiling mode. Only explicit shapes will help here.
//
OverlapDistributionParams getActivationOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                        ArrayRef<int64_t> activationTensorNumTiles);

// In case of output producing overlap lines with DPU
// we need to take into account all the consumer requirements
// when it comes to kernel, pad and stride.
//
// For the best handling, to provide the output which can service all consumers
// without spilling be required, we will need the support of precomputed shapes/offsets
// per cluster.
// This is because of the cases when different ops may have the maximum requirements
// on different clusters. In which case, there's no singular way with current distributed
// infrastructure to represent a mixed tiling mode. Only explicit shapes will help here.
//
OverlapDistributionParams getOutputOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                    ArrayRef<int64_t> outputTensorNumTiles,
                                                    vpux::NDTypeInterface outputType);

}  // namespace VPU
}  // namespace vpux

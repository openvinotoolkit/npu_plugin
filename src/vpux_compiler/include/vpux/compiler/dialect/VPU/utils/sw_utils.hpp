//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/ops.hpp"

namespace vpux {
namespace VPU {

VPU::DistributionMode getSWInputTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                       VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);

// General method of SW to get input tensor distribution mode
// For these SW Op, there are no specific inputs like 'auto_broadcast' or 'attribution'.
// There are examples for SW that can not use this general method:
//  - Multiply with SOH/SOK, the input with 'auto_broadcast' should be set to 'DUPLICATED' mode.
//  - InterpolateOp with SOHOverlapped, the input is sizes/scales/axes attribution should be set to 'DUPLICATED' mode.
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::MultiClusterStrategy strategy);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::InterpolateOp interpolateOp,
                                                       VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::MultiplyOp multiplyOp, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);

SmallVector<int64_t> getSWInputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
// General method of SW to get input tensor number tiles
// For these SW Op, there are no specific inputs like 'auto_broadcast' or 'attribution'.
// There are examples for SW that can not use this general method:
//  - Multiply with SOH/SOK, the input with 'auto_broadcast' should be set to [1, 1, 1, 1].
//  - InterpolateOp with SOHOverlapped, the input is sizes/scales/axes attribution should be set to [1, 1, 1, 1].
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::InterpolateOp interpolateOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::MultiplyOp multiplyOp, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);

//
// SameInOutDefaultDimsOrder
//

mlir::LogicalResult verifySameInOutDefaultDimsOrder(mlir::Operation* op);
void inferLayoutInfoSameInOutDefaultDimsOrder(IE::LayerLayoutInfo& info);

mlir::LogicalResult verifySameAnyDimsOrder(mlir::Operation* op);
void inferLayoutInfoSameAnyDimsOrder(IE::LayerLayoutInfo& info);

mlir::LogicalResult verifySameInOutSpecificDimsOrder(mlir::Operation* op, ArrayRef<DimsOrder> supportedLayouts);
void inferLayoutInfoSameInOutSpecificDimsOrder(IE::LayerLayoutInfo& info, ArrayRef<DimsOrder> supportedLayouts);

mlir::LogicalResult verifyReduceLayoutInfo(mlir::Operation* op);
void inferReduceLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info);

}  // namespace VPU
}  // namespace vpux

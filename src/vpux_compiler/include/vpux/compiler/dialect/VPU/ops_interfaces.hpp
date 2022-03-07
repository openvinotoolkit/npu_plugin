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

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPU {

//
// SparseOpInterface
//

bool supportsSparseInputs(mlir::Operation* op);
bool supportsSparseOutputs(mlir::Operation* op);
bool supportsSparseData(mlir::Operation* op);

//
// NCEOpInterface
//

namespace details {

mlir::LogicalResult validatePrecisionForNCE(mlir::Operation* op);
mlir::LogicalResult validateWorkloadsRegion(mlir::Location loc, mlir::Region& workloads);

mlir::Operation* addWorkload(mlir::Region& workloads, mlir::OpBuilder& builder, mlir::Location loc, ShapeRef offsets,
                             ShapeRef sizes, PaddingAttr pad, MPEMode mpeMode, mlir::IntegerAttr clusterId);

}  // namespace details

}  // namespace VPU
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPU/generated/ops_interfaces.hpp.inc>

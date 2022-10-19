//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/type_interfaces.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/Support/FormatVariadic.h>

namespace vpux {
namespace VPU {

class PaddingAttr;

}  // namespace VPU
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPU/generated/attributes/enums.hpp.inc>
#include <vpux/compiler/dialect/VPU/generated/attributes/structs.hpp.inc>

namespace vpux {
namespace VPU {

//
// Run-time resources
//

StringLiteral getMemoryDerateAttrName();
StringLiteral getMemoryBandwidthAttrName();
StringLiteral getProcessorFrequencyAttrName();

uint32_t getMaxDPUClusterNum(ArchKind arch);
uint32_t getMaxDPUClusterNum(mlir::Operation* op);
uint32_t getMaxDMAPorts(ArchKind arch);

Byte getTotalCMXSize(mlir::Operation* op);
Byte getTotalCMXSize(mlir::ModuleOp module);

//
// ArchKind
//

void setArch(mlir::ModuleOp module, ArchKind kind, Optional<int> numOfDPUGroups = None,
             Optional<int> numOfDMAPorts = None);
ArchKind getArch(mlir::Operation* op);

//
// CompilationMode
//

void setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode);
CompilationMode getCompilationMode(mlir::Operation* op);

//
// PaddingAttr
//

PaddingAttr getPaddingAttr(mlir::MLIRContext* ctx, int64_t left, int64_t right, int64_t top, int64_t bottom);
PaddingAttr getPaddingAttr(mlir::MLIRContext* ctx, ArrayRef<int64_t> padsBegin, ArrayRef<int64_t> padsEnd);
PaddingAttr getPaddingAttr(mlir::MLIRContext* ctx, const PadInfo& pad);

PadInfo toPadInfo(PaddingAttr attr);

//
// PPETaskAttr
//

PPETaskAttr getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode);

PPETaskAttr getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow, int64_t clampHigh,
                           int64_t lreluMult, int64_t lreluShift);

PPETaskAttr getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow, int64_t clampHigh,
                           int64_t lreluMult, int64_t lreluShift, ArrayRef<int64_t> quantMult,
                           ArrayRef<int64_t> quantShift, int64_t quantPostShift);

PPETaskAttr getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow, int64_t clampHigh,
                           int64_t lreluMult, int64_t lreluShift, ArrayRef<double> quantScale);

PPETaskAttr getPPETaskAttr(mlir::MLIRContext* ctx, VPU::PPEMode mode, int64_t clampLow, int64_t clampHigh,
                           int64_t lreluMult, int64_t lreluShift, ArrayRef<int64_t> quantMult,
                           ArrayRef<int64_t> quantShift, int64_t quantPostShift, ArrayRef<int64_t> in1QuantMult,
                           ArrayRef<int64_t> in2QuantMult);

VPU::PPEMode getPPEMode(VPU::EltwiseType type);

//
// DistributedTensorAttr
//

mlir::LogicalResult verify(FuncRef<mlir::InFlightDiagnostic()> emitError, DistributedTensorAttr distributedAttr,
                           ArrayRef<int64_t> shape);
mlir::LogicalResult areDistributionModesCompatible(DistributionMode sourceMode, DistributionMode targetMode);
mlir::LogicalResult areDistributionNumClustersCompatible(mlir::IntegerAttr sourceNumClusters,
                                                         mlir::IntegerAttr targetNumClusters);
mlir::LogicalResult areDistributionAttrsCompatible(DistributedTensorAttr sourceAttr, DistributedTensorAttr targetAttr);
SmallVector<Shape> getPerClusterComputeShapes(ShapeRef shapeRef, DistributedTensorAttr distributionAttr);
SmallVector<Shape> getPerClusterComputeShapeOffsets(ShapeRef shapeRef, DistributedTensorAttr distributionAttr);
SmallVector<PadInfo> getPerClusterPadding(DistributedTensorAttr distributionAttr);
SmallVector<StridedShape> getPerClusterStridedShapes(ShapeRef shape, StridesRef strides, DimsOrder dimsOrder,
                                                     DistributedTensorAttr distributionAttr);

}  // namespace VPU
}  // namespace vpux

//
// Copyright 2020 Intel Corporation.
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

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

//
// Generated
//

#include <vpux/compiler/dialect/VPUIPRegMapped/generated/attributes/enums.hpp.inc>

#include <mlir/IR/BuiltinOps.h>

namespace vpux {
namespace VPUIPRegMapped {

//
// MemoryLocation utilities
//

PhysicalMemory getPhysicalMemory(MemoryLocation location);
mlir::FailureOr<PhysicalMemory> getPhysicalMemory(mlir::MemRefType memref);

MemoryLocation getDefaultMemoryLocation(PhysicalMemory location);
mlir::FailureOr<VPUIPRegMapped::MemoryLocation> getMemoryLocation(mlir::MemRefType memref);

bool isMemoryCompatible(MemoryLocation location, mlir::MemRefType memref);

CompilationMode getCompilationMode(mlir::Operation* op);
void setCompilationMode(mlir::ModuleOp module, CompilationMode compilationMode);

}  // namespace VPUIPRegMapped
}  // namespace vpux

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

#include "vpux/compiler/dialect/VPUIP/blob_writer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {
namespace VPURT {

using MemoryEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;
void getTaskEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects);

}  // namespace VPURT
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPURT/generated/ops_interfaces.hpp.inc>

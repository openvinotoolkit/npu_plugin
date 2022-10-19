//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux {
namespace VPUIP {

constexpr int64_t DMA_MAX_NUMBER_PLANES = 256;
constexpr int64_t PER_PERMUTE_MAX_DMA_NUMBER = 8;
constexpr int64_t PERMUTE_DMA_MAX_LENTH = 256;

SmallVector<Shape> getDMASubShape(ShapeRef shape);
Optional<Shape> getPermuteDMAOutputShape(NDTypeInterface inType, NDTypeInterface outType, DimsOrder memPerm,
                                         vpux::Logger log);
Optional<SmallVector<Shape>> getPermuteDMASubShapes(VPUIP::PermuteDMAOp permuteOp, vpux::Logger log);
Optional<SmallVector<Shape>> getPermuteDMASubShapes(VPUIP::PermuteUPAOp permuteUPAOp, vpux::Logger log);
int64_t getDstStride(SmallVector<Shape> subShapes);
bool isBeneficialForUsingDMA(mlir::Operation* op, vpux::Logger log);
bool isCombineAtFront(ShapeRef shape, DimsOrder order);

}  // namespace VPUIP
}  // namespace vpux

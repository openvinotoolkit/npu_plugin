//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/attributes/structs.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

namespace NCESparsity {

// base_ptr is 9bits size
const int MAX_BASE_POINTER_VALUE = 511;
// Max number of distinct base pointers within one Storage Element Table. Should be equal as number of clusters
const int MAX_DISTINCT_BASE_PTRS = 4;

constexpr int32_t SPARSITY_PTR_WHEN_NO_SPARSITY = 0xFFFFFF;

using BiasConverterCb = int32_t (*)(double);
using PPEConverterCb = int32_t (*)(uint8_t, uint16_t, double, mlir::Type);

extern const EnumMap<ArchKind, PPEConverterCb> ppeConvertersMap;
extern const EnumMap<ArchKind, BiasConverterCb> biasConvertersMap;

int32_t toHex(double realVal);

enum class Mode { CM_CONV, DW_CONV, POOL };

int64_t getBitPatternSize(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType, int64_t IC);

int64_t getActivationWindowSize(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType, int64_t IC);

Shape inferActivationWindowShape(int64_t fakeSparsitySize);
Shape inferActivationWindowShape(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType, int64_t IC);

std::vector<uint8_t> getFakeSparsity(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType, int64_t IC);

int32_t getWeightPtrStep(mlir::Value weights, mlir::Value activationWindow);

std::vector<int32_t> getWeightsTable(mlir::Type inElemType, mlir::Type outElemType, Optional<int32_t> weightPtrOffset,
                                     int32_t weightPtrStep, Optional<int32_t> sparsityPtrOffset,
                                     int32_t sparsityPtrStep, ArchKind arch, int64_t OC,
                                     mlir::Type weightsElemType = nullptr, Const::ContentAttr bias = nullptr,
                                     VPU::PPETaskAttr ppe = nullptr, vpux::IE::PostOp postOpAttr = nullptr);

SmallVector<int32_t> getInstructionListTable(ArrayRef<int> rangeAttr, ArrayRef<int> shiftAttr, ArrayRef<int> biasAttr,
                                             const int32_t size);

Shape inferWeightsTableShape(int64_t OC);

mlir::FailureOr<SmallVector<double>> getRescaledBias(Const::ContentAttr biasAttr, mlir::Type inElemType,
                                                     mlir::Type filterElemType, int64_t OC);

}  // namespace NCESparsity

}  // namespace VPU
}  // namespace vpux

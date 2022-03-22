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
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

namespace NCESparsity {

constexpr int32_t SPARSITY_PTR_WHEN_NO_SPARSITY = 0xFFFFFF;

using BiasConverterCb = int32_t (*)(double);
using PPEConverterCb = int32_t (*)(uint8_t, uint16_t, double, mlir::Type);

extern const EnumMap<ArchKind, PPEConverterCb> ppeConvertersMap;
extern const EnumMap<ArchKind, BiasConverterCb> biasConvertersMap;

enum class Mode { CM_CONV, DW_CONV, POOL };

int64_t getBitPatternSize(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType, int64_t IC);

int64_t getActivationWindowSize(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType, int64_t IC);

Shape inferActivationWindowShape(int64_t fakeSparsitySize);
Shape inferActivationWindowShape(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType, int64_t IC);

std::vector<uint8_t> getFakeSparsity(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType, int64_t IC);

int32_t getWeightPtrStep(mlir::Value weights, mlir::Value activationWindow);

std::vector<int32_t> getWeightsTable(mlir::Type inElemType, mlir::Type outElemType, Optional<int32_t> weightPtrOffset,
                                     int32_t weightPtrStep, Optional<int32_t> sparsityPtrOffset, ArchKind arch,
                                     int64_t OC, mlir::Type weightsElemType = nullptr,
                                     Const::ContentAttr bias = nullptr, VPU::PPETaskAttr ppe = nullptr);

SmallVector<int32_t> getInstructionListTable(ArrayRef<int> rangeAttr, ArrayRef<int> shiftAttr, ArrayRef<int> biasAttr,
                                             const int32_t size);

Shape inferWeightsTableShape(int64_t OC);

}  // namespace NCESparsity

}  // namespace VPU
}  // namespace vpux

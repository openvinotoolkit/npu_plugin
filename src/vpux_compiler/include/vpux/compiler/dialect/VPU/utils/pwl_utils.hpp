//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

namespace vpux {
namespace VPU {

struct QuantInfo {
    double rMin;
    double rMax;
    double scale;
    int64_t zeroPoint;
    int64_t postShift;
};

struct PwlQuantReqs {
    QuantInfo input;
    QuantInfo output;
};

extern const EnumMap<VPU::PPEMode, PwlQuantReqs> pwlQuantReqs;

PwlQuantReqs getPwlQuantReqs(VPU::PPEMode ppeType);

int64_t getPwlPostShift(const VPU::PPEMode ppeType);
int64_t getPwlClamp(const mlir::Type inElemType, const mlir::Type outElemType, const VPU::PPEMode ppeType,
                    const bool getMin);

}  // namespace VPU
}  // namespace vpux

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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux {
namespace VPUIP {

class NCEInvariant final {
public:
    static constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

public:
    static mlir::LogicalResult verifyOp(mlir::Operation* op, Logger log = Logger::global());
};

}  // namespace VPUIP
}  // namespace vpux
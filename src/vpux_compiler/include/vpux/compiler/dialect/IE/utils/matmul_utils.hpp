//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BlockAndValueMapping.h>

namespace vpux {
namespace IE {
SmallVector<mlir::Operation*> getMatMulParentOps(IE::MatMulOp origOp);
bool checkPermuteMatMulPattern(IE::MatMulOp);
size_t getShapeSize(vpux::NDTypeInterface type);
SmallVector<int64_t, 2> getKernelFactors(int64_t total);
}  // namespace IE
}  // namespace vpux

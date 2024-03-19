//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux {
namespace VPU {

//
// ShaveKernelInfo
//

class ShaveKernelInfo {
public:
    ShaveKernelInfo(mlir::Operation* op): _swOp(op) {
    }

    virtual ~ShaveKernelInfo() = default;

    // Get the vector size used in specific kernel implementation
    virtual Bit getShaveVectorSize() const = 0;

protected:
    mlir::Operation* _swOp;
};

}  // namespace VPU
}  // namespace vpux

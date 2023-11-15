//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace vpux {
namespace VPURegMapped {

//
// Passes
//

// pass object stores callable, so it cannot llvm::function_ref
using UpperBoundCallable = std::function<size_t(VPURegMapped::TaskType, VPURegMapped::IndexType)>;

class ResolveTaskLocationPass : public vpux::FunctionPass {
public:
    using vpux::FunctionPass::FunctionPass;

protected:
    void safeRunOnFunc() final;
    template <typename Content>
    using MetadataBuffersContainer =
            llvm::SmallVector<llvm::DenseMap<VPURegMapped::TaskType, llvm::SmallVector<Content>>>;

    MetadataBuffersContainer<size_t> _metadataBuffersSizes;

private:
    MetadataBuffersContainer<llvm::SmallVector<mlir::Value>> _metadataBuffers;
};

}  // namespace VPURegMapped
}  // namespace vpux

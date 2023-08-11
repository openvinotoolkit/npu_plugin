//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/pipelines.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace vpux {

//
// ReferenceSWOptions30XX
//

struct ReferenceSWOptions30XX final : public ReferenceSWOptions<ReferenceSWOptions30XX> {};

void buildReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions30XX& options,
                                  Logger log = Logger::global());
void buildEMUReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions30XX& options,
                                     Logger log = Logger::global());

//
// ReferenceHWOptions30XX
//

struct ReferenceHWOptions30XX final : public ReferenceHWOptions<ReferenceHWOptions30XX> {
    BoolOption enableWeightsSparsity{*this, "enable-weights-sparsity", llvm::cl::desc("Enable weights sparsity"),
                                     llvm::cl::init(false)};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};
};

void buildReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions30XX& options,
                                  Logger log = Logger::global());

void buildEMUReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions30XX& options,
                                     Logger log = Logger::global());

//
// DefaultHWOptions30XX
//

struct DefaultHWOptions30XX final : public DefaultHWOptions<DefaultHWOptions30XX> {
    BoolOption enableWeightsSparsity{*this, "enable-weights-sparsity", llvm::cl::desc("Enable weights sparsity"),
                                     llvm::cl::init(false)};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};
};

void buildShaveCodeGenPipeline30XX(mlir::OpPassManager& pm, Logger log = Logger::global());

void buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions30XX& options,
                                Logger log = Logger::global());

void buildEMUDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions30XX& options,
                                   Logger log = Logger::global());

}  // namespace vpux

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace vpux {

//
// IPipelineRegister
//

class IPipelineRegister {
public:
    virtual void registerPipelines() = 0;

    virtual ~IPipelineRegister() = default;
};

//
// createPipelineRegister
//

std::unique_ptr<IPipelineRegister> createPipelineRegister(VPU::ArchKind archKind);

}  // namespace vpux

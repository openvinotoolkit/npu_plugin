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
// IPassesRegistry
//

class IPassesRegistry {
public:
    virtual void registerPasses() = 0;

    virtual ~IPassesRegistry() = default;
};

//
// createPassesRegistry
//

std::unique_ptr<IPassesRegistry> createPassesRegistry(VPU::ArchKind archKind);

}  // namespace vpux

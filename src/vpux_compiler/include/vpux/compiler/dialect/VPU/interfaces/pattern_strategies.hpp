//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux::VPU {

/*
   Class for getting specific patterns
*/
class IGreedilyPatternStrategy {
public:
    virtual ~IGreedilyPatternStrategy() = default;

    virtual void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const = 0;
};

}  // namespace vpux::VPU

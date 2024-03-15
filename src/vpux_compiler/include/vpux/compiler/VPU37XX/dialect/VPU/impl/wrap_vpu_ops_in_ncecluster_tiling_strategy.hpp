//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/pattern_strategies.hpp"

#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux::VPU::arch37xx {

/*
   Class for getting WrapVPUOpsInNCEClusterTilingStrategy patterns for VPU37XX
*/
class WrapVPUOpsInNCEClusterTilingStrategy : public IGreedilyPatternStrategy {
public:
    WrapVPUOpsInNCEClusterTilingStrategy(bool enableExplicitDistributedTensorAttr)
            : _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr) {
    }
    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override;

private:
    bool _enableExplicitDistributedTensorAttr = false;
};

}  // namespace vpux::VPU::arch37xx

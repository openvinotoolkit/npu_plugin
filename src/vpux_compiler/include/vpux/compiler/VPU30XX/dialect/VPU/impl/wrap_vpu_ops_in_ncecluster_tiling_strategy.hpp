//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/pattern_strategies.hpp"

namespace vpux::VPU::arch30xx {

/*
   Class for getting WrapVPUOpsInNCEClusterTilingStrategy patterns for VPU30XX
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

}  // namespace vpux::VPU::arch30xx

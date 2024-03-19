//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux::VPU {

//
// Strategy
//

class Strategy final {
public:
    Strategy(VPU::MultiClusterStrategy mcStrategy, mlir::ArrayAttr tilingStrategy,
             TilingMode mode = TilingMode::ISOLATED)
            : _mcStrategy(mcStrategy), _tilingStrategy(tilingStrategy), _mode(mode) {
    }

    VPU::MultiClusterStrategy getMCStrategy() const {
        return _mcStrategy;
    }

    mlir::ArrayAttr getTilingStrategy() const {
        return _tilingStrategy;
    }

    TilingMode getTilingMode() const {
        return _mode;
    }

    bool operator==(const Strategy& o) const {
        return (_mcStrategy == o._mcStrategy) && (_tilingStrategy == o._tilingStrategy) && (_mode == o._mode);
    }

    bool operator!=(const Strategy& o) const {
        return !((*this) == o);
    }

private:
    VPU::MultiClusterStrategy _mcStrategy;
    mlir::ArrayAttr _tilingStrategy;
    TilingMode _mode;
};

}  // namespace vpux::VPU

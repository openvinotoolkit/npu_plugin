//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/wrap_vpu_ops_in_ncecluster_tiling_strategy_getter.hpp"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include "vpux/compiler/VPU30XX/dialect/VPU/impl/wrap_vpu_ops_in_ncecluster_tiling_strategy.hpp"
#include "vpux/compiler/VPU37XX/dialect/VPU/impl/wrap_vpu_ops_in_ncecluster_tiling_strategy.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/logger.hpp"

using namespace vpux::VPU;

std::unique_ptr<IGreedilyPatternStrategy> vpux::VPU::createWrapVPUOpsInNCEClusterTilingStrategyGetter(
        mlir::func::FuncOp funcOp, bool enableExplicitDistributedTensorAttr) {
    const auto arch = getArch(funcOp);

    switch (arch) {
    case ArchKind::VPUX30XX: {
        return std::make_unique<arch30xx::WrapVPUOpsInNCEClusterTilingStrategy>(enableExplicitDistributedTensorAttr);
    }
    case ArchKind::VPUX37XX: {
        return std::make_unique<arch37xx::WrapVPUOpsInNCEClusterTilingStrategy>(enableExplicitDistributedTensorAttr);
    }
    case ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unable to get WrapVPUOpsInNCEClusterTilingStrategy for arch {0}", arch);
    }
    }
}

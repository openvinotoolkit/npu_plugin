//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/shave_kernel_info.hpp"
#include "vpux/compiler/VPU37XX/dialect/VPU/impl/shave_kernel_info.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

std::unique_ptr<VPU::ShaveKernelInfo> VPU::getShaveKernelInfo(mlir::Operation* op) {
    const auto arch = VPU::getArch(op);
    switch (arch) {
    case VPU::ArchKind::VPUX37XX: {
        return std::make_unique<VPU::arch37xx::ShaveKernelInfo>(op);
    }
    case VPU::ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
    }
}

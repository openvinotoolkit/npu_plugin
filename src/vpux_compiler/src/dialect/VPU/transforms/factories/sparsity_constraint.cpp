//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPU/impl/sparsity_constraint.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPU::SparsityConstraint VPU::getSparsityConstraint(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
    case VPU::ArchKind::VPUX37XX: {
        return VPU::arch30xx::SparsityConstraint{};
    }
    case VPU::ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
    }
}

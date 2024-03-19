//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/sparsity_constraint.hpp"

namespace vpux {
namespace VPU {

VPU::SparsityConstraint getSparsityConstraint(VPU::ArchKind arch);

}  // namespace VPU
}  // namespace vpux

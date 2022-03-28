//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/attributes.hpp"

#include <vpu_cost_model.h>

#include <memory>

namespace vpux {
namespace VPU {

std::shared_ptr<VPUNN::VPUCostModel> createCostModel(ArchKind arch);

}  // namespace VPU
}  // namespace vpux

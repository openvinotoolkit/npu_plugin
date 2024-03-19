//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux {

VPU::ArchKind getArchKind(const Config& config);
VPU::CompilationMode getCompilationMode(const Config& config);
std::optional<int> getNumberOfDPUGroups(const Config& config);
std::optional<int> getNumberOfDMAEngines(const Config& config);

}  // namespace vpux

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"

namespace vpux {

VPU::ArchKind getArchKind(const Config& config);
VPU::CompilationMode getCompilationMode(const Config& config);
Optional<int> getNumberOfDPUGroups(const Config& config);
Optional<int> getNumberOfDMAEngines(const Config& config);
Optional<int> getDDRHeapSize(const Config& config);

}  // namespace vpux

//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/passes.hpp"

namespace vpux::VPU::arch30xx {
void buildIncrementalPipeline(mlir::OpPassManager& pm, const VPU::TilingOptions& options,
                              Logger log = Logger::global());

void registerVPUPipelines();
}  // namespace vpux::VPU::arch30xx

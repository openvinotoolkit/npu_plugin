//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

#include <transformations/utils/utils.hpp>

#include <flatbuffers/flatbuffers.h>

namespace vpux {
namespace VPUIP {

flatbuffers::DetachedBuffer exportToBlob(mlir::ModuleOp module, mlir::TimingScope& rootTiming,
                                         const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                         const std::vector<std::shared_ptr<const ov::Node>>& results,
                                         Logger log = Logger::global());

}  // namespace VPUIP
}  // namespace vpux

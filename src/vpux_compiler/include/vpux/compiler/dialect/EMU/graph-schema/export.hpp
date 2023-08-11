//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"
#include "vpux_compiler.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

#include <flatbuffers/flatbuffers.h>

namespace vpux {
namespace EMU {

flatbuffers::DetachedBuffer exportToBlob(mlir::ModuleOp module, mlir::TimingScope& rootTiming,
                                         const std::vector<PreProcessInfo>& preprocessInfo,
                                         const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                         const std::vector<std::shared_ptr<const ov::Node>>& results,
                                         Logger log = Logger::global());

}  // namespace EMU
}  // namespace vpux

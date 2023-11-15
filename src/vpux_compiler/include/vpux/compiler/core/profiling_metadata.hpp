//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"

namespace vpux {

flatbuffers::DetachedBuffer buildProfilingMetaMI37XX(IE::CNNNetworkOp netOp, mlir::func::FuncOp funcOp, Logger _log);

flatbuffers::DetachedBuffer buildProfilingMetaVPURT(IE::CNNNetworkOp netOp, mlir::func::FuncOp funcOp, Logger _log);

};  // namespace vpux

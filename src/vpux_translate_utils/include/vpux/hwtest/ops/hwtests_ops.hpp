//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"

namespace vpux {
namespace hwtest {

void buildActShaveTask(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                       Logger& log, SmallVector<mlir::Type> inputTypes, vpux::VPURT::DeclareBufferOp inputCMX,
                       vpux::VPURT::DeclareBufferOp outputCMX, mlir::ValueRange waitBarrier,
                       mlir::ValueRange updateBarrier, size_t cluster = 0, size_t unit = 0);

}  // namespace hwtest
}  // namespace vpux

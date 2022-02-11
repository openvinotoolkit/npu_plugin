//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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

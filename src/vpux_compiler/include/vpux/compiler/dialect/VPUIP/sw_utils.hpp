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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

namespace vpux {
namespace VPUIP {

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module,
                                          mlir::StringRef builtInFunctionName, mlir::ArrayRef<mlir::Type> inputTypes,
                                          mlir::StringRef kernelEntryName, mlir::StringRef  kernelSourceFileName,
                                          const vpux::Logger& log);

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, IERT::LayerOpInterface origOp,
                                          const IERT::KernelInfo& kernelInfo, const Logger& log);

void initSwKernel(vpux::VPUIP::SwKernelOp swKernelOp, mlir::ValueRange inputs, mlir::ValueRange outputBuffs,
                  mlir::ArrayRef<mlir::Attribute> args, const vpux::Logger& log);

}  // namespace VPUIP
}  // namespace vpux

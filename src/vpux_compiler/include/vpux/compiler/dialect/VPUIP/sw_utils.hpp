//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

namespace vpux {
namespace VPUIP {

// TODO: E60214, need support more sw kernel task type. Currently only enable MVN
const SmallVector<StringLiteral> SW_KERNELS_SUPPORTING_TILING = {"singleShaveMVN"};
const SmallVector<StringLiteral> SW_KERNELS_SUPPORTING_STRIDE = {"singleShaveMVN"};

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, mlir::StringRef builtInFunctionName,
                                          const ArrayRef<mlir::Type> inputTypes, mlir::StringRef kernelEntryName,
                                          mlir::StringRef kernelSourceFileName, const vpux::Logger& log);

void createRuntimeKernelDefinition(mlir::ModuleOp module, const Logger& log);

void initSwKernel(vpux::VPUIP::SwKernelOp swKernelOp, mlir::ValueRange inputs, mlir::ValueRange outputBuffs,
                  mlir::ArrayRef<mlir::Attribute> args, const vpux::Logger& log);

void initSwKernel(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRunOp, const vpux::Logger& log);

bool isSwOpTilingSupported(VPU::SWOpInterface swOp);
SmallString getSwKernelEntryName(VPUIP::SwKernelOp swKernelOp);
bool isSwKernelTilingSupported(VPUIP::SwKernelOp swKernelOp);
bool isStridedDataAccessSupported(VPUIP::SwKernelOp swKernelOp);

}  // namespace VPUIP
}  // namespace vpux

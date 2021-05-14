//
// Copyright Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/PassManager.h>

namespace vpux {

void registerPipelines();

//
// Compiles IE Network in Reference mode (SW only execution).
//
// This pipeline performs full IR lowering from IE Dialect to VPUIP Dialect (using intermediate lowerings).
// It uses simple SW implementation for layer operations without any optimizations.
//

void buildReferenceModePipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildHardwareModePipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

}  // namespace vpux

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/Dialect.h>

namespace vpux::VPU::arch30xx {

void registerLayerWithPostOpModelInterface(mlir::DialectRegistry& registry);
void registerLayoutInfoOpInterfaces(mlir::DialectRegistry& registry);

}  // namespace vpux::VPU::arch30xx

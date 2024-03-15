//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"

#pragma once

namespace vpux {

namespace arch30xx {

//
// registerBufferizableOpInterfaces
//

void registerBufferizableOpInterfaces(mlir::DialectRegistry& registry);

}  // namespace arch30xx

}  // namespace vpux

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/interfaces_registry.hpp"
#include <mlir/IR/DialectRegistry.h>

#include "vpux/compiler/VPU30XX/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPUIP/ops_interfaces.hpp"

namespace vpux {

void InterfacesRegistry30XX::registerInterfaces(mlir::DialectRegistry& registry) {
    VPU::arch30xx::registerLayoutInfoOpInterfaces(registry);
    VPUIP::arch30xx::registerAlignedChannelsOpInterfaces(registry);
}

}  // namespace vpux

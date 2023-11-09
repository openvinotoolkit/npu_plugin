//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU37XX/interfaces_registry.hpp"
#include <mlir/IR/DialectRegistry.h>

#include "vpux/compiler/VPU37XX/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/VPU37XX/dialect/VPUIP/ops_interfaces.hpp"

namespace vpux {

void InterfacesRegistry37XX::registerInterfaces(mlir::DialectRegistry& registry) {
    VPU::arch37xx::registerLayoutInfoOpInterfaces(registry);
    VPUIP::arch37xx::registerAlignedChannelsOpInterfaces(registry);
}

}  // namespace vpux

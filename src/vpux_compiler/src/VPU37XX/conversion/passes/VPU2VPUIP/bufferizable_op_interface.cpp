//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU37XX/conversion/passes/VPU2VPUIP/bufferizable_op_interface.hpp"

//
// registerBufferizableOpInterfaces
//

void vpux::arch37xx::registerBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    vpux::registerFuncAndReturnBufferizableOpInterfaces(registry);
    vpux::registerSoftwareLayerBufferizableOpInterfaces(registry);
    vpux::registerVpuNceBufferizableOpInterfaces(registry);
}

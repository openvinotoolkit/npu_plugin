//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU37XX/passes_register.hpp"
#include "vpux/compiler/VPU37XX/conversion.hpp"
#include "vpux/compiler/VPU37XX/dialect/IE/passes.hpp"
#include "vpux/compiler/VPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/VPU37XX/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/VPU37XX/dialect/VPURT/passes.hpp"

using namespace vpux;

//
// PassesRegistry37XX::registerPasses
//

void PassesRegistry37XX::registerPasses() {
    vpux::arch37xx::registerConversionPasses();
    vpux::IE::arch37xx::registerIEPasses();
    vpux::VPU::arch37xx::registerVPUPasses();
    vpux::VPUIP::arch37xx::registerVPUIPPasses();
    vpux::VPURT::arch37xx::registerVPURTPasses();
}

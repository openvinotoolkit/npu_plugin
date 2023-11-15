//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/interfaces_registry.hpp"

#include "vpux/compiler/VPU30XX/interfaces_registry.hpp"
#include "vpux/compiler/VPU37XX/interfaces_registry.hpp"

#include <memory>

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {

//
// createInterfaceRegistry
//

std::unique_ptr<IInterfaceRegistry> createInterfacesRegistry(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return std::make_unique<InterfacesRegistry30XX>();
    case VPU::ArchKind::VPUX37XX:
        return std::make_unique<InterfacesRegistry37XX>();
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

}  // namespace vpux

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <memory>
#include "vpux/compiler/dialect/VPU/attributes.hpp"

namespace vpux {

//
// IInterfaceRegister
//

class IInterfaceRegistry {
public:
    virtual void registerInterfaces(mlir::DialectRegistry& registry) = 0;
    virtual ~IInterfaceRegistry() = default;
};

//
// createInterface
//

std::unique_ptr<IInterfaceRegistry> createInterfacesRegistry(VPU::ArchKind arch);

}  // namespace vpux

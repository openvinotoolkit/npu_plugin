//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/interfaces_registry.hpp"

namespace vpux {

//
// IntefacesRegistry30XX
//

class InterfacesRegistry30XX final : public IInterfaceRegistry {
public:
    void registerInterfaces(mlir::DialectRegistry& registry) override;
};

}  // namespace vpux

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/passes_register.hpp"

namespace vpux {

//
// PassesRegistry30XX
//

class PassesRegistry30XX final : public IPassesRegistry {
public:
    void registerPasses() override;
};

}  // namespace vpux

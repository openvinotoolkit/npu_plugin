//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux_compiler.hpp"

namespace vpux {
namespace VPUIP {

class NetworkDescription final : public INetworkDescription {
public:
    explicit NetworkDescription(std::vector<char> blob);
};

}  // namespace VPUIP
}  // namespace vpux

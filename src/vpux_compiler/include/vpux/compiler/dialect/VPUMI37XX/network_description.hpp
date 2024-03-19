//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"
#include "vpux_compiler.hpp"

namespace vpux {
namespace VPUMI37XX {

class NetworkDescription final : public INetworkDescription {
public:
    explicit NetworkDescription(std::vector<char> blob);
};

}  // namespace VPUMI37XX
}  // namespace vpux

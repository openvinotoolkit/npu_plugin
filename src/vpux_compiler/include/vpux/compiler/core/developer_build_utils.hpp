//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/string_ref.hpp"

namespace vpux {

void parseEnv(StringRef envVarName, std::string& var);
void parseEnv(StringRef envVarName, bool& var);

}  // namespace vpux

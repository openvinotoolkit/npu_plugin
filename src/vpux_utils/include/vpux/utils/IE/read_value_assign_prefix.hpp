//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// Prefix for ReadValue and Assign operations in IE dialect VPUX compiler.
//

#pragma once

#include <string>

namespace vpux {

#define READVALUE_PREFIX std::string("vpux_ie_read_value_")
#define ASSIGN_PREFIX std::string("vpux_ie_assign_")

}  // namespace vpux

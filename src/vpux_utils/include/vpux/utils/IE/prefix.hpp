//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <string>

namespace vpux {

constexpr char LOCATION_LAYER_TYPE_PREFIX[] = "t";

//
// Prefix for ReadValue and Assign operations in IE dialect VPUX compiler.
//
#define READVALUE_PREFIX std::string("vpux_ie_read_value_")
#define ASSIGN_PREFIX std::string("vpux_ie_assign_")

inline const bool isStateInputName(const std::string& name) {
    return name.find(READVALUE_PREFIX) != std::string::npos;
}
inline const bool isStateOutputName(const std::string& name) {
    return name.find(ASSIGN_PREFIX) != std::string::npos;
}

}  // namespace vpux

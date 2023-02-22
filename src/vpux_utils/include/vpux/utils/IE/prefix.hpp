//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <string>

namespace vpux {

// This char is a separator between original layer name provided in xml
// and metadata added by the compiler.
// It is crucial to provide layer names matching the original model in xml.
// This symbol must be unique in layer name.
constexpr char ORIGINAL_NAME_SEPARATOR = '?';

//
// Prefix for ReadValue and Assign operations in IE dialect VPUX compiler.
//
#define READVALUE_PREFIX std::string("vpux_ie_read_value_")
#define ASSIGN_PREFIX std::string("vpux_ie_assign_")

}  // namespace vpux

//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/opsets/opset.hpp>
#include "vpux/utils/core/error.hpp"
namespace vpux {

//
// tryStrToInt
//
uint32_t tryStrToInt(const std::string& strVersion);

//
// extractOpsetVersion
//
uint32_t extractOpsetVersion();

}  // namespace vpux

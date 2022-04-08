//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux_private_config.hpp"

#include "vpux/utils/core/string_ref.hpp"

namespace vpux {
namespace IMD {

bool platformSupported(InferenceEngine::VPUXConfigParams::VPUXPlatform platform);
StringRef getAppName(InferenceEngine::VPUXConfigParams::VPUXPlatform platform);

}  // namespace IMD
}  // namespace vpux

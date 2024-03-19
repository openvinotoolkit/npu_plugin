//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux_private_properties.hpp"

#include "vpux/utils/core/string_ref.hpp"

namespace vpux {

bool platformSupported(InferenceEngine::VPUXConfigParams::VPUXPlatform platform);
StringRef getAppName(InferenceEngine::VPUXConfigParams::VPUXPlatform platform);

}  // namespace vpux

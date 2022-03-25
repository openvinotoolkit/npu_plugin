//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "ie_remote_context.hpp"
#include <RemoteMemory.h>
#include "vpux/vpux_plugin_params.hpp"

namespace RemoteBlob_Helper {
    static InferenceEngine::ParamMap wrapRemoteMemFDToMap(const VpuxRemoteMemoryFD remoteMemoryFD) {
        return {{InferenceEngine::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD}};
    }
}

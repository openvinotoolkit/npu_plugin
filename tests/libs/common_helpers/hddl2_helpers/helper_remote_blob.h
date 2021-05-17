//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <RemoteMemory.h>
#include "vpux/vpux_plugin_params.hpp"
#include "ie_remote_context.hpp"

namespace RemoteBlob_Helper {
    static InferenceEngine::ParamMap wrapRemoteMemFDToMap(const VpuxRemoteMemoryFD remoteMemoryFD) {
        return {{InferenceEngine::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD}};
    }
}

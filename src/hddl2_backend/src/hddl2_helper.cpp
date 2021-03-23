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

#include "hddl2_helper.h"

#include "hddl2/hddl2_params.hpp"
#include "hddl2_exceptions.h"

namespace vpux {
namespace HDDL2 {

namespace IE = InferenceEngine;
HddlUnite::RemoteMemory::Ptr getRemoteMemoryFromParams(const InferenceEngine::ParamMap& params) {
    HddlUnite::RemoteMemory::Ptr remoteMemory;
    if (params.empty()) {
        IE_THROW() << PARAMS_ERROR_str << "Param map for allocator is empty.";
    }

    // Check that it's really contains required params
    const auto remote_memory_iter = params.find(IE::HDDL2_PARAM_KEY(REMOTE_MEMORY));
    if (remote_memory_iter == params.end()) {
        IE_THROW() << PARAMS_ERROR_str << "Param map does not contain remote memory file descriptor information";
    }
    try {
        remoteMemory = remote_memory_iter->second.as<HddlUnite::RemoteMemory::Ptr>();
    } catch (...) {
        IE_THROW() << CONFIG_ERROR_str << "Remote memory param have incorrect type information";
    }
    return remoteMemory;
}

}  // namespace HDDL2
}  // namespace vpux

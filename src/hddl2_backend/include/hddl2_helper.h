//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/logger.hpp"

// IE
#include <ie_remote_context.hpp>

// Low-level
#include <RemoteMemory.h>
#include <WorkloadContext.h>

namespace vpux {
namespace hddl2 {

HddlUnite::RemoteMemory* getRemoteMemoryFromParams(const InferenceEngine::ParamMap& params);
int32_t getRemoteMemoryFDFromParams(const InferenceEngine::ParamMap& params);
std::shared_ptr<InferenceEngine::TensorDesc> getOriginalTensorDescFromParams(const InferenceEngine::ParamMap& params);
WorkloadID getWorkloadIDFromParams(const InferenceEngine::ParamMap& params);
void setUniteLogLevel(Logger logger);
std::map<uint32_t, std::string> getSwDeviceIdNameMap();
std::string getSwDeviceIdFromName(const std::string& devName);

}  // namespace hddl2
}  // namespace vpux

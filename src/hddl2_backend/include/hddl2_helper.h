//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once
// IE
#include <ie_remote_context.hpp>
// Plugin
#include <vpux_config.hpp>
// Low-level
#include <RemoteMemory.h>
#include <WorkloadContext.h>

namespace vpux {
namespace hddl2 {

HddlUnite::RemoteMemory* getRemoteMemoryFromParams(const InferenceEngine::ParamMap& params);
int32_t getRemoteMemoryFDFromParams(const InferenceEngine::ParamMap& params);
std::shared_ptr<InferenceEngine::TensorDesc> getOriginalTensorDescFromParams(const InferenceEngine::ParamMap& params);
WorkloadID getWorkloadIDFromParams(const InferenceEngine::ParamMap& params);
void setUniteLogLevel(const vpu::LogLevel logLevel, const vpu::Logger::Ptr logger = nullptr);
std::map<uint32_t, std::string> getSwDeviceIdNameMap();
std::string getSwDeviceIdFromName(const std::string& devName);

}  // namespace hddl2
}  // namespace vpux

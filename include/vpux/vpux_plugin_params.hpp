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

/**
 * @brief A header for properties of shared device contexts and shared device memory blobs
 * for VPUX plugin
 * 
 * @deprecated Configuration API v1.0 would be deprecated in 2023.1 release.
 * It was left due to backward compatibility needs.
 * As such usage of this version of API is discouraged.
 * Prefer Configuration API v2.0.
 *
 * @file vpux_plugin_params.hpp
 */
#pragma once

#include <string>

using VpuxRemoteMemoryFD = int;
using VpuxHandleParam = void*;
using VpuxOffsetParam = size_t;

namespace InferenceEngine {
namespace VpuxContextParams {

/**
 * @def VPUX_PARAM_KEY(name)
 * @brief Shortcut for defining configuration keys
 * Configuration API v1.0
 */
#define VPUX_PARAM_KEY(name) VpuxContextParams::PARAM_##name

#ifndef DECLARE_PARAM_KEY_IMPL
#define DECLARE_PARAM_KEY_IMPL(...)
#endif

/**
 * @def DECLARE_VPUX_PARAM_KEY(name, ...)
 * @brief Shortcut for defining object parameter keys
 * Configuration API v1.0
 */
#define DECLARE_VPUX_PARAM_KEY(name, ...)        \
    static constexpr auto PARAM_##name = #name; \
    DECLARE_PARAM_KEY_IMPL(name, __VA_ARGS__)

/**
 * @brief Remote memory file descriptor
 * Configuration API v1.0
 */
DECLARE_VPUX_PARAM_KEY(REMOTE_MEMORY_FD, VpuxRemoteMemoryFD);

/**
 * @brief Memory handle
 * Configuration API v1.0
 */
DECLARE_VPUX_PARAM_KEY(MEM_HANDLE, VpuxHandleParam);

/**
 * @brief Memory offset to map physical address properly
 * Configuration API v1.0
 */
DECLARE_VPUX_PARAM_KEY(MEM_OFFSET, VpuxOffsetParam);

/**
 * @brief VPU device ID
 * Configuration API v1.0
 */
DECLARE_VPUX_PARAM_KEY(DEVICE_ID, std::string);

/**
 * @brief HDDLUnite Workload context id
 * Configuration API v1.0
 */
DECLARE_VPUX_PARAM_KEY(WORKLOAD_CONTEXT_ID, uint64_t);

}  // namespace VpuxContextParams
}  // namespace InferenceEngine

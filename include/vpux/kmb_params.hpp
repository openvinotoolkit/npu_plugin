//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

// TODO This file has been kept for backward compatibility
// Remove it in future releases after full transition to VPUX_PARAM
// [Track number: E#12122]

/**
 * @brief A header for properties of shared device contexts and shared device memory blobs
 * for KMB plugin
 *
 * @file kmb_params.hpp
 * @deprecated Use vpux_plugin_params.hpp instead
 */
#pragma once

#include <string>

using KmbRemoteMemoryFD = int;
using KmbHandleParam = void*;
using KmbOffsetParam = size_t;

namespace InferenceEngine {
namespace KmbContextParams {
/**
 * @def KMB_PARAM_KEY(name)
 * @brief Shortcut for defining configuration keys
 * @deprecated Use VPUX_PARAM_KEY instead
 * Configuration API v1.0
 */
#define KMB_PARAM_KEY(name) KmbContextParams::PARAM_##name

#ifndef DECLARE_PARAM_KEY_IMPL
#define DECLARE_PARAM_KEY_IMPL(...)
#endif

/**
 * @def DECLARE_KMB_PARAM_KEY(name, ...)
 * @brief Shortcut for defining object parameter keys
 * @deprecated Use DECLARE_VPUX_PARAM_KEY instead
 * Configuration API v1.0
 */
#define DECLARE_KMB_PARAM_KEY(name, ...)        \
    static constexpr auto PARAM_##name = #name; \
    DECLARE_PARAM_KEY_IMPL(name, __VA_ARGS__)

/**
 * @brief Remote memory file descriptor
 * @deprecated Use VPUX_PARAM(REMOTE_MEMORY_FD) instead
 * Configuration API v1.0
 */
DECLARE_KMB_PARAM_KEY(REMOTE_MEMORY_FD, KmbRemoteMemoryFD);

/**
 * @brief Remote memory handle
 * @deprecated Use VPUX_PARAM(MEM_HANDLE) instead
 * Configuration API v1.0
 */
DECLARE_KMB_PARAM_KEY(MEM_HANDLE, KmbHandleParam);

/**
 * @brief Remote memory offset to map physical address properly
 * @deprecated Use VPUX_PARAM(MEM_OFFSET) instead
 * Configuration API v1.0
 */
DECLARE_KMB_PARAM_KEY(MEM_OFFSET, KmbOffsetParam);

/**
 * @brief VPU device ID
 * @deprecated Use VPUX_PARAM(DEVICE_ID) instead
 * Configuration API v1.0
 */
DECLARE_KMB_PARAM_KEY(DEVICE_ID, std::string);
}  // namespace KmbContextParams
}  // namespace InferenceEngine

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

/**
 * @brief A header for properties of shared device contexts and shared device memory blobs
 * for KMB plugin
 *
 * @file kmb_params.hpp
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
 */
#define KMB_PARAM_KEY(name) KmbContextParams::PARAM_##name

#ifndef DECLARE_PARAM_KEY_IMPL
#define DECLARE_PARAM_KEY_IMPL(...)
#endif

/**
 * @def DECLARE_KMB_PARAM_KEY(name, ...)
 * @brief Shortcut for defining object parameter keys
 */
#define DECLARE_KMB_PARAM_KEY(name, ...)        \
    static constexpr auto PARAM_##name = #name; \
    DECLARE_PARAM_KEY_IMPL(name, __VA_ARGS__)

/**
 * @brief Remote memory file descriptor
 */
DECLARE_KMB_PARAM_KEY(REMOTE_MEMORY_FD, KmbRemoteMemoryFD);

/**
 * @brief Remote memory handle
 */
DECLARE_KMB_PARAM_KEY(MEM_HANDLE, KmbHandleParam);

/**
 * @brief Remote memory offset to map physical address properly
 */
DECLARE_KMB_PARAM_KEY(MEM_OFFSET, KmbOffsetParam);

/**
 * @brief VPU device ID
 */
DECLARE_KMB_PARAM_KEY(DEVICE_ID, std::string);
}  // namespace KmbContextParams
}  // namespace InferenceEngine

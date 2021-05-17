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
 * for VPUX plugin
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
 */
#define VPUX_PARAM_KEY(name) VpuxContextParams::PARAM_##name

#ifndef DECLARE_PARAM_KEY_IMPL
#define DECLARE_PARAM_KEY_IMPL(...)
#endif

/**
 * @def DECLARE_VPUX_PARAM_KEY(name, ...)
 * @brief Shortcut for defining object parameter keys
 */
#define DECLARE_VPUX_PARAM_KEY(name, ...)        \
    static constexpr auto PARAM_##name = #name; \
    DECLARE_PARAM_KEY_IMPL(name, __VA_ARGS__)

/**
 * @brief Remote memory file descriptor
 */
DECLARE_VPUX_PARAM_KEY(REMOTE_MEMORY_FD, VpuxRemoteMemoryFD);

/**
 * @brief Memory handle
 */
DECLARE_VPUX_PARAM_KEY(MEM_HANDLE, VpuxHandleParam);

/**
 * @brief Memory offset to map physical address properly
 */
DECLARE_VPUX_PARAM_KEY(MEM_OFFSET, VpuxOffsetParam);

/**
 * @brief VPU device ID
 */
DECLARE_VPUX_PARAM_KEY(DEVICE_ID, std::string);

/**
 * @brief HDDLUnite Workload context id
 */
DECLARE_VPUX_PARAM_KEY(WORKLOAD_CONTEXT_ID, uint64_t);

}  // namespace VpuxContextParams
}  // namespace InferenceEngine

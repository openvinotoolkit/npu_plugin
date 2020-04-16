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
 * for HDDL2 plugin
 *
 * @file hddl2_params.hpp
 */
#pragma once

#include <string>
#include <ie_blob.h>

namespace InferenceEngine {
namespace HDDL2ContextParams {
/**
 * @def HDDL2_PARAM_KEY(name)
 * @brief Shortcut for defining configuration keys
 */
#define HDDL2_PARAM_KEY(name) HDDL2ContextParams::PARAM_##name

#ifndef DECLARE_PARAM_KEY_IMPL
# define DECLARE_PARAM_KEY_IMPL(...)
#endif

/**
 * @def DECLARE_HDDL2_PARAM_KEY(name, ...)
 * @brief Shortcut for defining object parameter keys
 */
#define DECLARE_HDDL2_PARAM_KEY(name, ...)         \
    static constexpr auto PARAM_##name = #name;    \
    DECLARE_PARAM_KEY_IMPL(name, __VA_ARGS__)

    /**
     * @brief HDDLUnite Workload context id
     */
    DECLARE_HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID, uint64_t);

    /**
     * @brief HDDLUnite Remote memory file descriptor
     */
    DECLARE_HDDL2_PARAM_KEY(REMOTE_MEMORY_FD, uint64_t);

    /**
     * @brief Color format of remote memory
     */
    DECLARE_HDDL2_PARAM_KEY(COLOR_FORMAT, InferenceEngine::ColorFormat);

    /**
     * @brief ROI of blob
     */
    DECLARE_HDDL2_PARAM_KEY(ROI, InferenceEngine::ROI);
}  // namespace HDDL2ContextParams
}  // namespace InferenceEngine

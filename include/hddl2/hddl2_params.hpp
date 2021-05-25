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

// TODO This file has been kept for backward compatibility
// Remove it in future releases after full transition to VPUX_PARAM
// [Track number: E#12122]

/**
 * @brief A header for properties of shared device contexts and shared device memory blobs
 * for HDDL2 plugin
 *
 * @file hddl2_params.hpp
 * @deprecated Use vpux_plugin_params.hpp instead
 */
#pragma once

#include <string>
#include <ie_blob.h>

namespace InferenceEngine {
namespace HDDL2ContextParams {
/**
 * @def HDDL2_PARAM_KEY(name)
 * @brief Shortcut for defining configuration keys
 * @deprecated Use VPUX_PARAM_KEY instead
 */
#define HDDL2_PARAM_KEY(name) HDDL2ContextParams::PARAM_##name

#ifndef DECLARE_PARAM_KEY_IMPL
# define DECLARE_PARAM_KEY_IMPL(...)
#endif

/**
 * @def DECLARE_HDDL2_PARAM_KEY(name, ...)
 * @brief Shortcut for defining object parameter keys
 * @deprecated Use DECLARE_VPUX_PARAM_KEY instead
 */
#define DECLARE_HDDL2_PARAM_KEY(name, ...)         \
    static constexpr auto PARAM_##name = #name;    \
    DECLARE_PARAM_KEY_IMPL(name, __VA_ARGS__)

    /**
     * @brief HDDLUnite Workload context id
     * @deprecated Use VPUX_PARAM_KEY(WORKLOAD_CONTEXT_ID) instead
     */
    DECLARE_HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID, uint64_t);

    /**
     * @brief HDDLUnite Remote memory file descriptor
     * @deprecated Use VPUX_PARAM_KEY(REMOTE_MEMORY_FD) instead
     */
    DECLARE_HDDL2_PARAM_KEY(REMOTE_MEMORY, HddlUnite::RemoteMemory::Ptr);

    /**
     * @brief Color format of remote memory
     * @deprecated Currently unused, will be deleted soon
     */
    DECLARE_HDDL2_PARAM_KEY(COLOR_FORMAT, InferenceEngine::ColorFormat);

}  // namespace HDDL2ContextParams
}  // namespace InferenceEngine

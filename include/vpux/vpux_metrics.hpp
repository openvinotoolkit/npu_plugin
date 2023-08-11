//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "ie_plugin_config.hpp"

/**
 * @def VPUX_METRIC_KEY(name)
 * @brief Shortcut for defining VPUX Plugin specific metrics (metrics which are not shared across other plugins)
 */
#define VPUX_METRIC_KEY(name) METRIC_KEY(VPUX_##name)

namespace InferenceEngine {
namespace Metrics {
#define DECLARE_VPUX_METRIC_KEY(name, ...) DECLARE_METRIC_KEY(VPUX_##name, __VA_ARGS__)

/**
 * @brief Metric to get the name of used backend
 */
DECLARE_VPUX_METRIC_KEY(BACKEND_NAME, std::string);

/**
 * @brief Metric to get total size of available DDR memory
 *
 * Note: Queries driver when device is discrete; returns host memory size when device is integrated
 */
DECLARE_VPUX_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE, uint64_t);

}  // namespace Metrics
}  // namespace InferenceEngine

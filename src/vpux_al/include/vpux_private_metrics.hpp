//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

/**
 * @def VPUX_METRIC_KEY(name)
 * @brief Shortcut for defining VPUX Plugin metrics
 */
#define VPUX_METRIC_KEY(name) METRIC_KEY(VPUX_##name)

namespace InferenceEngine {
namespace Metrics {
#define DECLARE_VPUX_METRIC_KEY(name, ...) DECLARE_METRIC_KEY(VPUX_##name, __VA_ARGS__)

/**
 * @brief Metric to get the name of used backend
 */
DECLARE_VPUX_METRIC_KEY(BACKEND_NAME, std::string);

}  // namespace Metrics
}  // namespace InferenceEngine

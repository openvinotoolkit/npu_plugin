//
// Copyright 2021 Intel Corporation.
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

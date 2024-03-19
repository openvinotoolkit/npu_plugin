//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/algorithms/simulated_annealing.hpp"

namespace vpux {
namespace algorithm {
bool defaultStopCondition(size_t temperature) {
    return temperature <= 0;
}

void defaultTemperatureCallback(size_t& temperature) {
    --temperature;
}
}  // namespace algorithm
}  // namespace vpux

// Copyright 2021 (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"
#include "vpux_private_metrics.hpp"

std::string getBackendName(const ov::runtime::Core& core) {
    return core.get_metric("VPUX", VPUX_METRIC_KEY(BACKEND_NAME)).as<std::string>();
} 

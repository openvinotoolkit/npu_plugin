//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/parsed_config.hpp"

using namespace vpux;

//
// LAUNCH_MODE
//

StringLiteral vpux::IMD::stringifyEnum(LaunchMode val) {
    switch (val) {
    case LaunchMode::MoviSim:
        return "MoviSim";
    default:
        return "<UNKNOWN>";
    }
}

IMD::LaunchMode vpux::IMD::LAUNCH_MODE::parse(StringRef val) {
    if (val == VPUX_IMD_CONFIG_VALUE(MOVI_SIM)) {
        return LaunchMode::MoviSim;
    }

    VPUX_THROW("Value '{0}' is not a valid VPUX_IMD_LAUNCH_MODE option", val);
}

//
// MV_RUN_TIMEOUT
//

std::chrono::seconds vpux::IMD::MV_RUN_TIMEOUT::defaultValue() {
    using namespace std::chrono_literals;
    static const auto RUN_TIMEOUT = std::chrono::duration_cast<std::chrono::seconds>(20min);
    return RUN_TIMEOUT;
}

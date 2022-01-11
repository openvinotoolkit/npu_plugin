//
// Copyright Intel Corporation.
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

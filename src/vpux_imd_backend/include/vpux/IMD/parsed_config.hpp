//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/IMD/private_config.hpp"

#include "vpux/utils/IE/config.hpp"

#include <chrono>

namespace vpux {
namespace IMD {

//
// MV_TOOLS_PATH
//

struct MV_TOOLS_PATH final : OptionBase<MV_TOOLS_PATH, std::string> {
    static StringRef key() {
        return VPUX_IMD_CONFIG_KEY(MV_TOOLS_PATH);
    }

    static StringRef envVar() {
        return "IE_VPUX_MV_TOOLS_PATH";
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// LAUNCH_MODE
//

enum class LaunchMode { Simulator, MoviDebug };

StringLiteral stringifyEnum(LaunchMode val);

struct LAUNCH_MODE final : OptionBase<LAUNCH_MODE, LaunchMode> {
    static StringRef key() {
        return VPUX_IMD_CONFIG_KEY(LAUNCH_MODE);
    }

    static StringRef envVar() {
        return "IE_VPUX_IMD_LAUNCH_MODE";
    }

    static LaunchMode parse(StringRef val);

    static LaunchMode defaultValue() {
        return LaunchMode::Simulator;
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

struct MV_RUN_TIMEOUT final : OptionBase<MV_RUN_TIMEOUT, std::chrono::seconds> {
    static StringRef key() {
        return VPUX_IMD_CONFIG_KEY(MV_RUN_TIMEOUT);
    }

    static StringRef envVar() {
        return "IE_MV_RUN_TIMEOUT";
    }

    static std::chrono::seconds defaultValue();

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

}  // namespace IMD
}  // namespace vpux

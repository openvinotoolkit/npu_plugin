//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/IE/config.hpp"

#include "vpual_private_config.hpp"
#include "vpual_private_properties.hpp"

namespace vpux {

//
// REPACK_INPUT_LAYOUT
//

struct VPUAL_REPACK_INPUT_LAYOUT final : OptionBase<VPUAL_REPACK_INPUT_LAYOUT, bool> {
    static StringRef key() {
        return ov::intel_vpux::repack_input_layout.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

}  // namespace vpux

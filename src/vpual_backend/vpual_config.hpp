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

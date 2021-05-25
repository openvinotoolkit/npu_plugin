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

#include "vpual_config.hpp"

#include <map>
#include <string>
#include <unordered_set>
#include <vpux/vpux_plugin_config.hpp>

#include "vpual_private_config.hpp"

namespace vpux {

VpualConfig::VpualConfig() {
    _runTimeOptions = merge(vpux::VPUXConfig::getRunTimeOptions(), {
                                                                       VPUX_VPUAL_CONFIG_KEY(REPACK_INPUT_LAYOUT),
                                                                       VPUX_VPUAL_CONFIG_KEY(INFERENCE_SHAVES),
                                                                   });
}

void VpualConfig::parse(const std::map<std::string, std::string>& config) {
    vpux::VPUXConfig::parse(config);

    setOption(_repackInputLayout, switches, config, VPUX_VPUAL_CONFIG_KEY(REPACK_INPUT_LAYOUT));
    setOption(_numberOfNnCoreShaves, config, VPUX_VPUAL_CONFIG_KEY(INFERENCE_SHAVES), parseInt);
    IE_ASSERT(0 <= _numberOfNnCoreShaves && _numberOfNnCoreShaves <= 16)
        << "VPUXConfig::parse attempt to set invalid number of shaves for NnCore: '" << _numberOfNnCoreShaves
        << "', valid numbers are from 0 to 16";
}

}  // namespace vpux

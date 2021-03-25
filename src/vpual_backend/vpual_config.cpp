//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpual_config.hpp"

#include <cpp_interfaces/exception2status.hpp>
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

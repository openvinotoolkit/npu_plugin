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
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/numeric.hpp>

#include "vpual_private_config.hpp"

namespace vpux {

const std::unordered_set<std::string>& VpualConfig::getRunTimeOptions() const {
    static const std::unordered_set<std::string> options =
        merge(vpux::VPUXConfig::getCompileOptions(), {
                                                         VPU_KMB_CONFIG_KEY(FORCE_NCHW_TO_NHWC),
                                                     });

    return options;
}

void VpualConfig::parse(const std::map<std::string, std::string>& config) {
    vpux::VPUXConfig::parse(config);

    setOption(_forceNCHWToNHWC, switches, config, VPU_KMB_CONFIG_KEY(FORCE_NCHW_TO_NHWC));
}

}  // namespace vpux

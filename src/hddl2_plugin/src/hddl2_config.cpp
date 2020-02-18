//
// Copyright 2019 Intel Corporation.
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

#include "hddl2_config.h"

#include <cpp_interfaces/exception2status.hpp>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/numeric.hpp>

using namespace vpu;

const std::unordered_set<std::string>& HDDL2Config::getCompileOptions() const {
    // TODO: Add new config header for HDDL2
    static const std::unordered_set<std::string> options =
        merge(MCMConfig::getCompileOptions(), {
                                                  VPU_KMB_CONFIG_KEY(PLATFORM),
                                              });
    return options;
}

void HDDL2Config::parse(const std::map<std::string, std::string>& config) {
    MCMConfig::parse(config);

    setOption(_platform, switches, config, VPU_KMB_CONFIG_KEY(PLATFORM));
}

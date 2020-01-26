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

#include "kmb_config.h"

#include <cpp_interfaces/exception2status.hpp>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/numeric.hpp>

using namespace vpu::KmbPlugin;

const std::unordered_set<std::string>& KmbConfig::getCompileOptions() const {
    static const std::unordered_set<std::string> options =
        merge(MCMConfig::getCompileOptions(), {
                                                  VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION),
                                                  VPU_KMB_CONFIG_KEY(PLATFORM),
                                              });

    return options;
}

const std::unordered_set<std::string>& KmbConfig::getRunTimeOptions() const {
    static const std::unordered_set<std::string> options =
        merge(MCMConfig::getCompileOptions(), {
                                                  VPU_KMB_CONFIG_KEY(KMB_EXECUTOR),
                                                  VPU_KMB_CONFIG_KEY(THROUGHPUT_STREAMS),
                                                  VPU_KMB_CONFIG_KEY(PREPROCESSING_SHAVES),
                                                  VPU_KMB_CONFIG_KEY(PREPROCESSING_LPI),
                                              });

    return options;
}

void KmbConfig::parse(const std::map<std::string, std::string>& config) {
    MCMConfig::parse(config);

    setOption(_useKmbExecutor, switches, config, VPU_KMB_CONFIG_KEY(KMB_EXECUTOR));

    setOption(_loadNetworkAfterCompilation, switches, config, VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION));

    setOption(_throghputStreams, config, VPU_KMB_CONFIG_KEY(THROUGHPUT_STREAMS), parseInt);

    setOption(_platform, switches, config, VPU_KMB_CONFIG_KEY(PLATFORM));

    setOption(_numberOfSIPPShaves, config, VPU_KMB_CONFIG_KEY(PREPROCESSING_SHAVES), parseInt);
    IE_ASSERT(_numberOfSIPPShaves > 0 && _numberOfSIPPShaves <= 16)
        << "KmbConfig::parse attempt to set invalid number of shaves for SIPP: '" << _numberOfSIPPShaves
        << "', valid numbers are from 1 to 16";

    setOption(_SIPPLpi, config, VPU_KMB_CONFIG_KEY(PREPROCESSING_LPI), parseInt);
    IE_ASSERT(0 < _SIPPLpi && _SIPPLpi <= 16 && isPowerOfTwo(_SIPPLpi))
        << "KmbConfig::parse attempt to set invalid lpi value for SIPP: '" << _SIPPLpi
        << "',  valid values are 1, 2, 4, 8, 16";
}

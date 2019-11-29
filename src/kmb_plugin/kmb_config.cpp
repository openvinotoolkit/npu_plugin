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

using namespace vpu::KmbPlugin;

KmbConfig::KmbConfig() {
    _config = {
#ifdef NDEBUG
        {CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_NONE)},
#else
        {CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_DEBUG)},
#endif
#ifdef ENABLE_VPUAL
        {VPU_KMB_CONFIG_KEY(KMB_EXECUTOR), CONFIG_VALUE(YES)},
#else
        {VPU_KMB_CONFIG_KEY(KMB_EXECUTOR), CONFIG_VALUE(NO)},
#endif
        {VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR_PATH), "mcm_config/target"},
        {VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR), "release_kmb"},
        {VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR_PATH), "mcm_config/compilation"},
        {VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR), "release_kmb"},
        {VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB), CONFIG_VALUE(YES)},
        {VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON), CONFIG_VALUE(YES)},
        {VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT), CONFIG_VALUE(NO)},
        {VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY), CONFIG_VALUE(NO)},
        {VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS_PATH), "."},
        {VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS), ""},
        {VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION), CONFIG_VALUE(NO)},
        {VPU_KMB_CONFIG_KEY(THROUGHPUT_STREAMS), "1"},
        {VPU_KMB_CONFIG_KEY(PREPROCESSING_SHAVES), "4"},
        {VPU_KMB_CONFIG_KEY(PLATFORM), "VPU_2490"},
    };
}

const std::unordered_set<std::string>& KmbConfig::getCompileOptions() const {
    static const std::unordered_set<std::string> options =
        merge(ParsedConfigBase::getCompileOptions(), {
                                                         VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR_PATH),
                                                         VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR),
                                                         VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR_PATH),
                                                         VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR),
                                                         VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB),
                                                         VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY),
                                                         VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON),
                                                         VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT),
                                                         VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS_PATH),
                                                         VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS),
                                                         VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION),
                                                         VPU_KMB_CONFIG_KEY(PLATFORM),
                                                     });

    return options;
}

const std::unordered_set<std::string>& KmbConfig::getRunTimeOptions() const {
    static const std::unordered_set<std::string> options =
        merge(ParsedConfigBase::getCompileOptions(), {
                                                         VPU_KMB_CONFIG_KEY(KMB_EXECUTOR),
                                                         VPU_KMB_CONFIG_KEY(THROUGHPUT_STREAMS),
                                                         VPU_KMB_CONFIG_KEY(PREPROCESSING_SHAVES),
                                                     });

    return options;
}

void KmbConfig::parse(const std::map<std::string, std::string>& config) {
    for (const auto& p : config) {
        _config[p.first] = p.second;
    }
    ParsedConfigBase::parse(config);

    std::istringstream strToNum(_config[VPU_KMB_CONFIG_KEY(PREPROCESSING_SHAVES)]);
    strToNum >> numberOfSIPPShaves;
    IE_ASSERT(numberOfSIPPShaves > 0 && numberOfSIPPShaves <= 16)
        << "KmbConfig::parse attempt to set invalid number of shaves for SIPP: '"
        << _config[VPU_KMB_CONFIG_KEY(PREPROCESSING_SHAVES)] << "', valid numbers are from 1 to 16";
}

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

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vpu/kmb_plugin_config.hpp>

using namespace vpu;
namespace IE = InferenceEngine;

const std::unordered_set<std::string>& HDDL2Config::getCompileOptions() const {
    // TODO: Add new config header for HDDL2
    static const std::unordered_set<std::string> options =
        merge(MCMConfig::getCompileOptions(), {
                                                  // TODO Just to avoid error, do nothing
                                                  VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION),
                                              });
    return options;
}

const std::unordered_set<std::string>& HDDL2Config::getRunTimeOptions() const {
    static const std::unordered_set<std::string> options =
        merge(ParsedConfigBase::getRunTimeOptions(), {
                                                         CONFIG_KEY(PERF_COUNT),
                                                         CONFIG_KEY(DEVICE_ID),
                                                     });

    return options;
}

void HDDL2Config::parse(const std::map<std::string, std::string>& config) {
    MCMConfig::parse(config);

    static const std::unordered_map<std::string, LogLevel> logLevels = {{CONFIG_VALUE(LOG_NONE), LogLevel::None},
        {CONFIG_VALUE(LOG_ERROR), LogLevel::Error}, {CONFIG_VALUE(LOG_WARNING), LogLevel::Warning},
        {CONFIG_VALUE(LOG_INFO), LogLevel::Info}, {CONFIG_VALUE(LOG_DEBUG), LogLevel::Debug},
        {CONFIG_VALUE(LOG_TRACE), LogLevel::Trace}};

    setOption(_device_id, config, CONFIG_KEY(DEVICE_ID));
    setOption(_logLevel, logLevels, config, CONFIG_KEY(LOG_LEVEL));
    setOption(_performance_counting, switches, config, CONFIG_KEY(PERF_COUNT));
}

LogLevel HDDL2Config::logLevel() const { return _logLevel; }

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
                                                         VPU_KMB_CONFIG_KEY(MCM_LOG_LEVEL),
                                                         VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION),
                                                         VPU_KMB_CONFIG_KEY(PLATFORM),
                                                         VPU_KMB_CONFIG_KEY(ELTWISE_SCALES_ALIGNMENT),
                                                         VPU_KMB_CONFIG_KEY(INPUT_SCALE_SHIFT_REMOVING),
                                                     });

    return options;
}

const std::unordered_set<std::string>& KmbConfig::getRunTimeOptions() const {
    static const std::unordered_set<std::string> options =
        merge(ParsedConfigBase::getCompileOptions(), {
                                                         VPU_KMB_CONFIG_KEY(KMB_EXECUTOR),
                                                         VPU_KMB_CONFIG_KEY(THROUGHPUT_STREAMS),
                                                         VPU_KMB_CONFIG_KEY(PREPROCESSING_SHAVES),
                                                         VPU_KMB_CONFIG_KEY(PREPROCESSING_LPI),
                                                     });

    return options;
}

void KmbConfig::parse(const std::map<std::string, std::string>& config) {
    static const std::unordered_map<std::string, LogLevel> logLevels = {{CONFIG_VALUE(LOG_NONE), LogLevel::None},
        {CONFIG_VALUE(LOG_ERROR), LogLevel::Error}, {CONFIG_VALUE(LOG_WARNING), LogLevel::Warning},
        {CONFIG_VALUE(LOG_INFO), LogLevel::Info}, {CONFIG_VALUE(LOG_DEBUG), LogLevel::Debug},
        {CONFIG_VALUE(LOG_TRACE), LogLevel::Trace}};

    ParsedConfigBase::parse(config);

    setOption(_useKmbExecutor, switches, config, VPU_KMB_CONFIG_KEY(KMB_EXECUTOR));

    setOption(_mcmLogLevel, logLevels, config, VPU_KMB_CONFIG_KEY(MCM_LOG_LEVEL));

    setOption(_mcmTargetDesciptorPath, config, VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR_PATH));
    setOption(_mcmTargetDesciptor, config, VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR));

    setOption(_mcmCompilationDesciptorPath, config, VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR_PATH));
    setOption(_mcmCompilationDesciptor, config, VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR));

    setOption(_mcmGenerateBlob, switches, config, VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB));
    setOption(_mcmGenerateJSON, switches, config, VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON));
    setOption(_mcmGenerateDOT, switches, config, VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT));

    setOption(_mcmParseOnly, switches, config, VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY));

    setOption(_mcmCompilationResultsPath, config, VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS_PATH));
    setOption(_mcmCompilationResults, config, VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS));

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

    setOption(_eltwiseScalesAlignment, switches, config, VPU_KMB_CONFIG_KEY(ELTWISE_SCALES_ALIGNMENT));
    setOption(_inputScaleShiftRemoving, switches, config, VPU_KMB_CONFIG_KEY(INPUT_SCALE_SHIFT_REMOVING));
}

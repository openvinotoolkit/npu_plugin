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

#include <file_utils.h>

#include <vpu/utils/numeric.hpp>
#include <vpux/vpux_compiler_config.hpp>
#include <vpux/vpux_plugin_config.hpp>
#include <vpux_config.hpp>

#include "vpux_private_config.hpp"

namespace IE = InferenceEngine;

vpux::VPUXConfig::VPUXConfig() {
    _compileOptions =
            merge(vpux::VPUXConfigBase::getCompileOptions(), {
                                                                     VPUX_CONFIG_KEY(PLATFORM),
                                                                     VPU_COMPILER_CONFIG_KEY(USE_NGRAPH_PARSER),
                                                             });
    _runTimeOptions = merge(vpux::VPUXConfigBase::getRunTimeOptions(), {
                                                                               CONFIG_KEY(PERF_COUNT),
                                                                               CONFIG_KEY(DEVICE_ID),
                                                                               VPUX_CONFIG_KEY(THROUGHPUT_STREAMS),
                                                                               KMB_CONFIG_KEY(THROUGHPUT_STREAMS),
                                                                               VPUX_CONFIG_KEY(PLATFORM),
                                                                               VPUX_CONFIG_KEY(GRAPH_COLOR_FORMAT),
                                                                               VPUX_CONFIG_KEY(CSRAM_SIZE),
                                                                               VPUX_CONFIG_KEY(USE_M2I),
                                                                               VPU_KMB_CONFIG_KEY(USE_M2I),
                                                                               VPUX_CONFIG_KEY(USE_SHAVE_ONLY_M2I),
                                                                               VPU_KMB_CONFIG_KEY(USE_SHAVE_ONLY_M2I),
                                                                               VPUX_CONFIG_KEY(USE_SIPP),
                                                                               VPU_KMB_CONFIG_KEY(USE_SIPP),
                                                                               VPUX_CONFIG_KEY(PREPROCESSING_SHAVES),
                                                                               VPUX_CONFIG_KEY(PREPROCESSING_LPI),
                                                                               VPUX_CONFIG_KEY(PREPROCESSING_PIPES),
                                                                               VPUX_CONFIG_KEY(EXECUTOR_STREAMS),
                                                                               VPU_KMB_CONFIG_KEY(EXECUTOR_STREAMS),
                                                                               VPUX_CONFIG_KEY(INFERENCE_TIMEOUT),
                                                                       });
}

void vpux::VPUXConfig::parseFrom(const vpux::VPUXConfig& other) {
    parse(other.getConfig());
}

void vpux::VPUXConfig::parseEnvironment() {
#ifndef NDEBUG
    if (const auto envVar = std::getenv("KMB_USE_LEGACY_PARSER")) {
        _useNGraphParser = !std::stoi(envVar);
    }
#endif
}

void vpux::VPUXConfig::parse(const std::map<std::string, std::string>& config) {
    vpux::VPUXConfigBase::parse(config);

    // Public options
    setOption(_performanceCounting, switches, config, CONFIG_KEY(PERF_COUNT));

    setOption(_deviceId, config, CONFIG_KEY(DEVICE_ID));
    setOption(_throughputStreams, config, VPUX_CONFIG_KEY(THROUGHPUT_STREAMS), parseInt);
    setOption(_throughputStreams, config, KMB_CONFIG_KEY(THROUGHPUT_STREAMS), parseInt);
    static const std::unordered_map<std::string, IE::VPUXConfigParams::VPUXPlatform> vpuxPlatform = {
            {VPUX_CONFIG_VALUE(AUTO), IE::VPUXConfigParams::VPUXPlatform::AUTO},
            {VPUX_CONFIG_VALUE(MA2490), IE::VPUXConfigParams::VPUXPlatform::MA2490},
            {VPUX_CONFIG_VALUE(MA2490_B0), IE::VPUXConfigParams::VPUXPlatform::MA2490_B0},
            {VPUX_CONFIG_VALUE(MA3100), IE::VPUXConfigParams::VPUXPlatform::MA3100},
            {VPUX_CONFIG_VALUE(MA3720), IE::VPUXConfigParams::VPUXPlatform::MA3720}};
    setOption(_platform, vpuxPlatform, config, VPUX_CONFIG_KEY(PLATFORM));

    // Private options
    setOption(_inferenceTimeoutMs, config, VPUX_CONFIG_KEY(INFERENCE_TIMEOUT), parseInt);
    setOption(_useNGraphParser, switches, config, VPU_COMPILER_CONFIG_KEY(USE_NGRAPH_PARSER));
    static const std::unordered_map<std::string, IE::ColorFormat> colorFormat = {
            {VPUX_CONFIG_VALUE(BGR), IE::ColorFormat::BGR},
            {VPUX_CONFIG_VALUE(RGB), IE::ColorFormat::RGB}};
    setOption(_graphColorFormat, colorFormat, config, VPUX_CONFIG_KEY(GRAPH_COLOR_FORMAT));
    setOption(_csramSize, config, VPUX_CONFIG_KEY(CSRAM_SIZE), parseInt);
    setOption(_useM2I, switches, config, VPUX_CONFIG_KEY(USE_M2I));
    setOption(_useM2I, switches, config, VPU_KMB_CONFIG_KEY(USE_M2I));
    setOption(_useSHAVE_only_M2I, switches, config, VPUX_CONFIG_KEY(USE_SHAVE_ONLY_M2I));
    setOption(_useSHAVE_only_M2I, switches, config, VPU_KMB_CONFIG_KEY(USE_SHAVE_ONLY_M2I));
    setOption(_useSIPP, switches, config, VPUX_CONFIG_KEY(USE_SIPP));
    setOption(_useSIPP, switches, config, VPU_KMB_CONFIG_KEY(USE_SIPP));
    setOption(_numberOfSIPPShaves, config, VPUX_CONFIG_KEY(PREPROCESSING_SHAVES), parseInt);
    IE_ASSERT(_numberOfSIPPShaves > 0 && _numberOfSIPPShaves <= 16)
            << "VPUXConfig::parse attempt to set invalid number of shaves for SIPP: '" << _numberOfSIPPShaves
            << "', valid numbers are from 1 to 16";
    setOption(_SIPPLpi, config, VPUX_CONFIG_KEY(PREPROCESSING_LPI), parseInt);
    IE_ASSERT(0 < _SIPPLpi && _SIPPLpi <= 16 && vpu::isPowerOfTwo(_SIPPLpi))
            << "VPUXConfig::parse attempt to set invalid lpi value for SIPP: '" << _SIPPLpi
            << "',  valid values are 1, 2, 4, 8, 16";
    setOption(_numberOfPPPipes, config, VPUX_CONFIG_KEY(PREPROCESSING_PIPES), parseInt);
    setOption(_executorStreams, config, VPUX_CONFIG_KEY(EXECUTOR_STREAMS), parseInt);
    setOption(_executorStreams, config, VPU_KMB_CONFIG_KEY(EXECUTOR_STREAMS), parseInt);

    parseEnvironment();
}

std::string vpux::getLibFilePath(const std::string& baseName) {
    return FileUtils::makeSharedLibraryName(InferenceEngine::getIELibraryPath(), baseName + IE_BUILD_POSTFIX);
}

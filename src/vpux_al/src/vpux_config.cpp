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

#include <file_utils.h>

#include <vpux/vpux_compiler_config.hpp>
#include <vpux/vpux_plugin_config.hpp>
#include <vpux_config.hpp>

namespace IE = InferenceEngine;

vpux::VPUXConfig::VPUXConfig() {
    _compileOptions = merge(vpux::VPUXConfigBase::getCompileOptions(), {
                                                                               VPUX_CONFIG_KEY(PLATFORM),
                                                                               VPUX_CONFIG_KEY(COMPILER_TYPE),
                                                                               VPUX_CONFIG_KEY(COMPILATION_MODE),
                                                                       });
    _runTimeOptions = merge(vpux::VPUXConfigBase::getRunTimeOptions(), {
                                                                               CONFIG_KEY(PERF_COUNT),
                                                                               CONFIG_KEY(DEVICE_ID),
                                                                               VPUX_CONFIG_KEY(THROUGHPUT_STREAMS),
                                                                               KMB_CONFIG_KEY(THROUGHPUT_STREAMS),
                                                                               VPUX_CONFIG_KEY(INFERENCE_SHAVES),
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
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    if (const auto env = std::getenv("IE_VPUX_COMPILER_TYPE")) {
        if (std::strcmp(env, VPUX_CONFIG_VALUE(MCM)) == 0) {
            _compilerType = IE::VPUXConfigParams::CompilerType::MCM;
        } else if (std::strcmp(env, VPUX_CONFIG_VALUE(MLIR)) == 0) {
            _compilerType = IE::VPUXConfigParams::CompilerType::MLIR;
        } else {
            IE_THROW() << "Invalid value "
                       << "\"" << env << "\""
                       << " for key IE_VPUX_COMPILER_TYPE environment variable";
        }
    }

    if (const auto env = std::getenv("IE_VPUX_COMPILATION_MODE")) {
        _compilationMode = env;
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
    setOption(_numberOfNnCoreShaves, config, VPUX_CONFIG_KEY(INFERENCE_SHAVES), parseInt);
    IE_ASSERT(0 <= _numberOfNnCoreShaves && _numberOfNnCoreShaves <= 16)
            << "VPUXConfig::parse attempt to set invalid number of shaves for NnCore: '" << _numberOfNnCoreShaves
            << "', valid numbers are from 0 to 16";
    static const std::unordered_map<std::string, IE::VPUXConfigParams::VPUXPlatform> vpuxPlatform = {
            {VPUX_CONFIG_VALUE(AUTO), IE::VPUXConfigParams::VPUXPlatform::AUTO},
            {VPUX_CONFIG_VALUE(VPU3400_A0), IE::VPUXConfigParams::VPUXPlatform::VPU3400_A0},
            {VPUX_CONFIG_VALUE(VPU3400), IE::VPUXConfigParams::VPUXPlatform::VPU3400},
            {VPUX_CONFIG_VALUE(VPU3700), IE::VPUXConfigParams::VPUXPlatform::VPU3700},
            {VPUX_CONFIG_VALUE(VPU3800), IE::VPUXConfigParams::VPUXPlatform::VPU3800},
            {VPUX_CONFIG_VALUE(VPU3900), IE::VPUXConfigParams::VPUXPlatform::VPU3900},
            {VPUX_CONFIG_VALUE(VPU3720), IE::VPUXConfigParams::VPUXPlatform::VPU3720}};
    setOption(_platform, vpuxPlatform, config, VPUX_CONFIG_KEY(PLATFORM));

    // Private options
    setOption(_inferenceTimeoutMs, config, VPUX_CONFIG_KEY(INFERENCE_TIMEOUT), parseInt);
    static const std::unordered_map<std::string, IE::ColorFormat> colorFormat = {
            {VPUX_CONFIG_VALUE(BGR), IE::ColorFormat::BGR},
            {VPUX_CONFIG_VALUE(RGB), IE::ColorFormat::RGB}};
    setOption(_graphColorFormat, colorFormat, config, VPUX_CONFIG_KEY(GRAPH_COLOR_FORMAT));
    setOption(_csramSize, config, VPUX_CONFIG_KEY(CSRAM_SIZE), parseInt);
    // Set maximum possible CSRAM size to 1 Gb
    const decltype(_csramSize) MAX_CSRAM_SIZE = 1024 * 1024 * 1024;
    IE_ASSERT(_csramSize >= -1 && _csramSize <= MAX_CSRAM_SIZE)
            << "VPUXConfig::parse attempt to set invalid CSRAM size in bytes: '" << _csramSize
            << "', valid values are -1, 0 and up to 1 Gb";
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
    static const std::unordered_map<std::string, IE::VPUXConfigParams::CompilerType> vpuxCompilerType = {
            {VPUX_CONFIG_VALUE(MCM), IE::VPUXConfigParams::CompilerType::MCM},
            {VPUX_CONFIG_VALUE(MLIR), IE::VPUXConfigParams::CompilerType::MLIR}};
    setOption(_compilerType, vpuxCompilerType, config, VPUX_CONFIG_KEY(COMPILER_TYPE));
    setOption(_compilationMode, config, VPUX_CONFIG_KEY(COMPILATION_MODE));

    parseEnvironment();
}

std::string vpux::getLibFilePath(const std::string& baseName) {
    return FileUtils::makePluginLibraryName(InferenceEngine::getIELibraryPath(), baseName + IE_BUILD_POSTFIX);
}

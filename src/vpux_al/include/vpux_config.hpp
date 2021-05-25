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

#pragma once

#include <vpux/vpux_plugin_config.hpp>

#include "vpux_config_base.hpp"
#include "vpux_private_config.hpp"

namespace vpux {

class VPUXConfig : public VPUXConfigBase {
public:
    VPUXConfig();

    // Public options
    bool performanceCounting() const {
        return _performanceCounting;
    }
    std::string deviceId() const {
        return _deviceId;
    }
    int throughputStreams() const {
        return _throughputStreams;
    }
    int numberOfNnCoreShaves() const {
        return _numberOfNnCoreShaves;
    }
    InferenceEngine::VPUXConfigParams::VPUXPlatform platform() const {
        return _platform;
    }

    // Private options
    InferenceEngine::ColorFormat graphColorFormat() const {
        return _graphColorFormat;
    }
    int32_t CSRAMSize() const {
        return _csramSize;
    }
    bool useM2I() const {
        return _useM2I;
    }
    bool useSHAVE_only_M2I() const {
        return _useSHAVE_only_M2I;
    }
    bool useSIPP() const {
        return _useSIPP;
    }
    int numberOfSIPPShaves() const {
        return _numberOfSIPPShaves;
    }
    int SIPPLpi() const {
        return _SIPPLpi;
    }
    int numberOfPPPipes() const {
        return _numberOfPPPipes;
    }
    int executorStreams() const {
        return _executorStreams;
    }
    uint32_t inferenceTimeoutMs() const noexcept {
        return _inferenceTimeoutMs;
    }
    InferenceEngine::VPUXConfigParams::CompilerType compilerType() const noexcept {
        return _compilerType;
    }
    const std::string& compilationMode() const {
        return _compilationMode;
    }

    void parseFrom(const VPUXConfig& other);

protected:
    void parse(const std::map<std::string, std::string>& config) override;

    // Public options
    bool _performanceCounting = false;
    std::string _deviceId = "VPU-0";
    int _throughputStreams = 2;
    int _numberOfNnCoreShaves = 0;
    InferenceEngine::VPUXConfigParams::VPUXPlatform _platform = InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO;
    int32_t _csramSize = -1;

    // Private options
    InferenceEngine::ColorFormat _graphColorFormat = InferenceEngine::ColorFormat::BGR;
    // FIXME: Likely has to be true by default as well.
    // NB.: Currently applies to the detection use-case only
    bool _useM2I = false;
    bool _useSHAVE_only_M2I = false;
    bool _useSIPP = true;
    int _numberOfSIPPShaves = 4;
    int _SIPPLpi = 8;
    int _numberOfPPPipes = 1;
    int _executorStreams = 1;
    // backend pull timeout - 5 seconds by default
    uint32_t _inferenceTimeoutMs = 5 * 1000;
    InferenceEngine::VPUXConfigParams::CompilerType _compilerType =
            InferenceEngine::VPUXConfigParams::CompilerType::MCM;

    std::string _compilationMode = "ReferenceSW";

private:
    void parseEnvironment();
};

std::string getLibFilePath(const std::string& baseName);

}  // namespace vpux

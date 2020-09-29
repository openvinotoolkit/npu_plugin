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

#pragma once

#include <ie_common.h>

#include <vpu/parsed_config_base.hpp>

namespace vpux {

class VPUXConfigBase : public vpu::ParsedConfigBase {
public:
    VPUXConfigBase();
    const std::map<std::string, std::string>& getConfig() const { return _config; }
    const std::unordered_set<std::string>& getCompileOptions() const override;
    const std::unordered_set<std::string>& getRunTimeOptions() const override;
    void expandSupportedOptions(const std::unordered_set<std::string>& options);

protected:
    void parse(const std::map<std::string, std::string>& config) override;
    std::unordered_set<std::string> _options;

private:
    std::map<std::string, std::string> _config;
};

class VPUXConfig : public VPUXConfigBase {
public:
    VPUXConfig();
    bool useNGraphParser() const { return _useNGraphParser; }

    int throughputStreams() const { return _throughputStreams; }
    int executorStreams() const { return _executorStreams; }

    const std::string& platform() const { return _platform; }

    int numberOfSIPPShaves() const { return _numberOfSIPPShaves; }

    int SIPPLpi() const { return _SIPPLpi; }

    // FIXME: drop SIPP from the method name
    InferenceEngine::ColorFormat outColorFmtSIPP() const { return _outColorFmtSIPP; }

    bool useSIPP() const { return _useSIPP; }

    bool useM2I() const { return _useM2I; }

    bool performanceCounting() const { return _performanceCounting; }

    std::string deviceId() const { return _deviceId; }

    void parseFrom(const VPUXConfig& other);

protected:
    void parse(const std::map<std::string, std::string>& config) override;

    bool _useNGraphParser = true;

    int _throughputStreams = 1;
    int _executorStreams = 1;

    std::string _platform = "VPU_2490";

    int _numberOfSIPPShaves = 4;
    int _SIPPLpi = 8;
    InferenceEngine::ColorFormat _outColorFmtSIPP = InferenceEngine::ColorFormat::BGR;
    bool _performanceCounting = false;
    bool _useSIPP = true;

    // FIXME: Likely has to be true by default as well.
    // NB.: Currently applies to the detection use-case only
    bool _useM2I = false;

    std::string _deviceId = "VPU-0";
};

}  // namespace vpux

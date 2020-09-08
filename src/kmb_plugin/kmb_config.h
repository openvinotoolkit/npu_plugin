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

#pragma once

#include <ie_common.h>

#include <map>
#include <string>
#include <unordered_set>
#include <vpux_config.hpp>

namespace vpu {
namespace KmbPlugin {

class KmbConfig final : public vpux::VPUXConfig {
public:
    bool useKmbExecutor() const { return _useKmbExecutor; }

    bool loadNetworkAfterCompilation() const { return _loadNetworkAfterCompilation; }

    int throughputStreams() const { return _throughputStreams; }

    const std::string& platform() const { return _platform; }

    int numberOfSIPPShaves() const { return _numberOfSIPPShaves; }

    int SIPPLpi() const { return _SIPPLpi; }

    // FIXME: drop SIPP from the method name
    InferenceEngine::ColorFormat outColorFmtSIPP() { return _outColorFmtSIPP; }

    bool forceNCHWToNHWC() const { return _forceNCHWToNHWC; }

    bool useSIPP() const { return _useSIPP; }

    bool useM2I() const { return _useM2I; }
    std::string deviceId() const { return _deviceId; }
    int preFetchSize() const { return _preFetchSize; }

public:
    void parse(const std::map<std::string, std::string>& config) override;

protected:
    const std::unordered_set<std::string>& getCompileOptions() const override;
    const std::unordered_set<std::string>& getRunTimeOptions() const override;

private:
#if defined(__arm__) || defined(__aarch64__)
    bool _useKmbExecutor = true;
#else
    bool _useKmbExecutor = false;
#endif

#ifdef __aarch64__
    bool _loadNetworkAfterCompilation = true;
#else
    bool _loadNetworkAfterCompilation = false;
#endif

    int _throughputStreams = 1;

    std::string _platform = "VPU_2490";

    int _numberOfSIPPShaves = 4;
    int _SIPPLpi = 8;
    InferenceEngine::ColorFormat _outColorFmtSIPP = InferenceEngine::ColorFormat::BGR;
    bool _forceNCHWToNHWC = false;
    // FIXME: have to be true, disabled due to not working vpu runtime
    // tracking number: h#18011604382
    bool _useSIPP = true;

    // FIXME: Likely has to be true by default as well.
    // NB.: Currently applies to the detection use-case only
    bool _useM2I = false;
    std::string _deviceId = "VPU-0";
    int _preFetchSize = 0;
};

}  // namespace KmbPlugin
}  // namespace vpu

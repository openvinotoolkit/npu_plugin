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
#include <mcm_config.h>

#include <map>
#include <string>
#include <unordered_set>

namespace vpu {
namespace KmbPlugin {

namespace ie = InferenceEngine;

class KmbConfig final : public MCMConfig {
public:
    bool useKmbExecutor() const { return _useKmbExecutor; }

    bool loadNetworkAfterCompilation() const { return _loadNetworkAfterCompilation; }

    int throghputStreams() const { return _throghputStreams; }

    const std::string& platform() const { return _platform; }

    int numberOfSIPPShaves() const { return _numberOfSIPPShaves; }

    int SIPPLpi() const { return _SIPPLpi; }

    InferenceEngine::ColorFormat outColorFmtSIPP() { return _outColorFmtSIPP; }

    bool forceNCHWToNHWC() { return _forceNCHWToNHWC; }

protected:
    const std::unordered_set<std::string>& getCompileOptions() const override;
    const std::unordered_set<std::string>& getRunTimeOptions() const override;
    void parse(const std::map<std::string, std::string>& config) override;

private:
#ifdef ENABLE_VPUAL
    bool _useKmbExecutor = true;
#else
    bool _useKmbExecutor = false;
#endif

#ifdef __aarch64__
    bool _loadNetworkAfterCompilation = true;
#else
    bool _loadNetworkAfterCompilation = false;
#endif

    int _throghputStreams = 1;

    std::string _platform = "VPU_2490";

    int _numberOfSIPPShaves = 4;
    int _SIPPLpi = 8;
    InferenceEngine::ColorFormat _outColorFmtSIPP = InferenceEngine::ColorFormat::BGR;
    bool _forceNCHWToNHWC = false;

private:
    static InferenceEngine::ColorFormat parseColorFormat(const std::string& src) {
        if (src == "RGB") {
            return ie::ColorFormat::RGB;
        } else if (src == "BGR") {
            return ie::ColorFormat::BGR;
        } else {
            THROW_IE_EXCEPTION << "Unsupported color format is passed.";
        }
    }
};

}  // namespace KmbPlugin
}  // namespace vpu

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

#include <map>
#include <string>
#include <unordered_set>
#include <vpu/parsed_config_base.hpp>

namespace vpu {
namespace KmbPlugin {

class KmbConfig final : public ParsedConfigBase {
public:
    bool useKmbExecutor() const { return _useKmbExecutor; }

    LogLevel mcmLogLevel() const { return _mcmLogLevel; }

    const std::string& mcmTargetDesciptorPath() const { return _mcmTargetDesciptorPath; }

    const std::string& mcmTargetDesciptor() const { return _mcmTargetDesciptor; }

    const std::string& mcmCompilationDesciptorPath() const { return _mcmCompilationDesciptorPath; }

    const std::string& mcmCompilationDesciptor() const { return _mcmCompilationDesciptor; }

    bool mcmGenerateBlob() const { return _mcmGenerateBlob; }

    bool mcmGenerateJSON() const { return _mcmGenerateJSON; }

    bool mcmGenerateDOT() const { return _mcmGenerateDOT; }

    bool mcmParseOnly() const { return _mcmParseOnly; }

    const std::string& mcmCompilationResultsPath() const { return _mcmCompilationResultsPath; }

    const std::string& mcmCompilationResults() const { return _mcmCompilationResults; }

    bool loadNetworkAfterCompilation() const { return _loadNetworkAfterCompilation; }

    int throghputStreams() const { return _throghputStreams; }

    const std::string& platform() const { return _platform; }

    int numberOfSIPPShaves() const { return _numberOfSIPPShaves; }

    int SIPPLpi() const { return _SIPPLpi; }

    bool eltwiseScalesAlignment() const { return _eltwiseScalesAlignment; }

    bool inputScaleShiftRemoving() const { return _inputScaleShiftRemoving; }

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

    LogLevel _mcmLogLevel = LogLevel::None;

    std::string _mcmTargetDesciptorPath = "mcm_config/target";
    std::string _mcmTargetDesciptor = "release_kmb";

    std::string _mcmCompilationDesciptorPath = "mcm_config/compilation";
    std::string _mcmCompilationDesciptor = "release_kmb";

    bool _mcmGenerateBlob = true;
    bool _mcmGenerateJSON = true;
    bool _mcmGenerateDOT = false;

    bool _mcmParseOnly = false;

    std::string _mcmCompilationResultsPath = ".";
    std::string _mcmCompilationResults = "";

    bool _loadNetworkAfterCompilation = false;
    int _throghputStreams = 1;

    std::string _platform = "VPU_2490";

    int _numberOfSIPPShaves = 4;
    int _SIPPLpi = 8;

    bool _eltwiseScalesAlignment = false;
    bool _inputScaleShiftRemoving = true;
};

}  // namespace KmbPlugin
}  // namespace vpu

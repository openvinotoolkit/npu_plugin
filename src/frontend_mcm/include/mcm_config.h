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

#include <map>
#include <string>
#include <unordered_set>
#include <vpu/parsed_config_base.hpp>
#include <vpu/vpu_compiler_config.hpp>

#include "frontend_private_config.hpp"

namespace vpu {

class MCMConfig : public ParsedConfigBase {
public:
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

    bool eltwiseScalesAlignment() const { return _eltwiseScalesAlignment; }

    bool concatScalesAlignment() const { return _concatScalesAlignment; }

    bool zeroPointsOnWeightsAlignment() const { return _zeroPointsOnWeightsAlignment; }

    const std::string& serializeCNNBeforeCompileFile() const { return _serializeCNNBeforeCompileFile; }

    bool useNGraphParser() const { return _useNGraphParser; }
    bool allowNCOutput() const { return _allowNCOutput; }
    bool allowFP32Output() const { return _allowFP32Output; }

protected:
    const std::unordered_set<std::string>& getCompileOptions() const override;
    void parse(const std::map<std::string, std::string>& config) override;

private:
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

    bool _eltwiseScalesAlignment = true;
    bool _concatScalesAlignment = true;
    bool _zeroPointsOnWeightsAlignment = true;
    std::string _serializeCNNBeforeCompileFile = "";

    bool _useNGraphParser = false;
    bool _allowNCOutput = false;
    bool _allowFP32Output = false;
};

}  // namespace vpu

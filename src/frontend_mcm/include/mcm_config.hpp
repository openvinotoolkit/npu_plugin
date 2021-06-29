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

#include <frontend_private_config.hpp>
#include <map>
#include <string>
#include <unordered_set>
#include <vpu/utils/logger.hpp>
#include <vpux_config.hpp>

namespace vpu {

class MCMConfig : public vpux::VPUXConfig {
public:
    LogLevel mcmLogLevel() const {
        return _mcmLogLevel;
    }

    const std::string& mcmTargetDesciptorPath() const {
        return _mcmTargetDesciptorPath;
    }

    const std::string& mcmTargetDesciptor() const {
        return _mcmTargetDesciptor;
    }

    const std::string& mcmCompilationDesciptorPath() const {
        return _mcmCompilationDesciptorPath;
    }

    const std::string& mcmCompilationDesciptor() const {
        return _mcmCompilationDesciptor;
    }

    bool mcmGenerateBlob() const {
        return _mcmGenerateBlob;
    }

    bool mcmGenerateJSON() const {
        return _mcmGenerateJSON;
    }

    bool mcmGenerateDOT() const {
        return _mcmGenerateDOT;
    }

    bool mcmParseOnly() const {
        return _mcmParseOnly;
    }

    const std::string& mcmCompilationResultsPath() const {
        return _mcmCompilationResultsPath;
    }

    const std::string& mcmCompilationResults() const {
        return _mcmCompilationResults;
    }

    bool eltwiseScalesAlignment() const {
        return _eltwiseScalesAlignment;
    }

    bool concatScalesAlignment() const {
        return _concatScalesAlignment;
    }

    bool zeroPointsOnWeightsAlignment() const {
        return _zeroPointsOnWeightsAlignment;
    }

    const std::string& serializeCNNBeforeCompileFile() const {
        return _serializeCNNBeforeCompileFile;
    }

    std::string customLayers() const {
        return _customLayers;
    }

    const std::string& mcmCompilationPassBanList() const {
        return _mcmCompilationPassBanList;
    }

    bool scaleFuseInput() const {
        return _scaleFuseInput;
    }

    bool referenceMode() const {
        return _referenceMode;
    }

    const std::unordered_set<std::string>& getCompileOptions() const override;

    bool allowNCHWLayoutForMcmModelInput() const {
        return _allowNCHWLayoutForMcmModelInput;
    }

    bool allowU8InputForFp16Models() const {
        return _allowU8InputForFp16Models;
    }

    bool scaleShiftFusing() const {
        return _scaleShiftFusing;
    }

    bool removePermuteNoOp() const {
        return _removePermuteNoOp;
    }

    bool allowPermuteND() const {
        return _allowPermuteND;
    }

    int numberOfClusters() const {
        return _numberOfClusters;
    }

    const std::string& layerSplitStrategies() const {
        return _layerSplitStrategies;
    }

    const std::string& layerStreamStrategies() const {
        return _layerStreamStrategies;
    }

    const std::string& layerSparsityStrategies() const {
        return _layerSparsityStrategies;
    }

    const std::string& layerLocationStrategies() const {
        return _layerLocationStrategies;
    }

    bool optimizeInputPrecision() const {
        return _optimizeInputPrecision;
    }

protected:
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

    std::string _customLayers = "";

    std::string _mcmCompilationPassBanList = "";

    bool _scaleFuseInput = false;

    bool _referenceMode = false;

    bool _allowNCHWLayoutForMcmModelInput = false;

    bool _allowU8InputForFp16Models = false;

    bool _scaleShiftFusing = true;

    bool _removePermuteNoOp = true;

    bool _allowPermuteND = false;

    int _numberOfClusters = 0;

    std::string _layerSplitStrategies = "";
    std::string _layerStreamStrategies = "";
    std::string _layerSparsityStrategies = "";
    std::string _layerLocationStrategies = "";

    bool _optimizeInputPrecision = true;
};

}  //  namespace vpu

//
// Copyright Intel Corporation.
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

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux_private_config.hpp"

#include <string>

namespace vpux {
namespace IMD {

class ExecutorImpl final : public Executor {
public:
    ExecutorImpl(InferenceEngine::VPUXConfigParams::VPUXPlatform platform, const NetworkDescription::Ptr& network,
                 const Config& config);

public:
    void setup(const InferenceEngine::ParamMap&) override;

    Executor::Ptr clone() const override;

    void push(const InferenceEngine::BlobMap& inputs) override;
    void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) override;

    void pull(InferenceEngine::BlobMap& outputs) override;

    bool isPreProcessingSupported(const PreprocMap&) const override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() override;

    InferenceEngine::Parameter getParameter(const std::string&) const override;

private:
    struct InferenceManagerDemo final {
        std::string elfFile;
        std::string runProgram;
        SmallVector<StringRef> runArgs;
        int64_t timeoutSec;
        std::string chipsetArg;
        std::string imdElfArg;
    };

    void parseAppConfig(InferenceEngine::VPUXConfigParams::VPUXPlatform platform, const Config& config);

    SmallString createTempWorkDir();
    void storeNetworkBlob(StringRef workDir);
    void storeNetworkInputs(StringRef workDir, const InferenceEngine::BlobMap& inputs);
    void runApp(StringRef workDir);
    void loadNetworkOutputs(StringRef workDir, const InferenceEngine::BlobMap& outputs);

private:
    NetworkDescription::Ptr _network;
    Logger _log;

    InferenceManagerDemo _app;

    InferenceEngine::BlobMap _inputs;
};

}  // namespace IMD
}  // namespace vpux

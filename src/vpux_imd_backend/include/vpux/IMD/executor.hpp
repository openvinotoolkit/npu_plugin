//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
        std::string chipsetVersion;
        std::string imdElf;
    };

    void parseAppConfig(InferenceEngine::VPUXConfigParams::VPUXPlatform platform, const Config& config);

    SmallString createTempWorkDir();
    void copyAppFile(StringRef workDir);
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

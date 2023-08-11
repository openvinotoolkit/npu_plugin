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
    struct InferenceManagerDemo;

    ExecutorImpl(InferenceEngine::VPUXConfigParams::VPUXPlatform platform, const NetworkDescription::Ptr& network,
                 const Config& config);

    Executor::Ptr clone() const override;

    NetworkDescription& getNetworkDesc() {
        return *_network.get();
    }

    InferenceManagerDemo& getApp() {
        return _app;
    }

    struct InferenceManagerDemo final {
        std::string elfFile;
        std::string runProgram;
        SmallVector<StringRef> runArgs;
        int64_t timeoutSec;
        std::string chipsetArg;
        std::string imdElfArg;
    };

private:
    std::string getMoviToolsPath(const Config& config);
    std::string getSimicsPath(const Config& config);
    void setMoviSimRunArgs(InferenceEngine::VPUXConfigParams::VPUXPlatform platform, const Config& config);
    void setMoviDebugRunArgs(InferenceEngine::VPUXConfigParams::VPUXPlatform platform, const Config& config);
    void setSimicsRunArgs(InferenceEngine::VPUXConfigParams::VPUXPlatform platform, const Config& config);

    bool isValidElfSignature(StringRef filePath);
    void parseAppConfig(InferenceEngine::VPUXConfigParams::VPUXPlatform platform, const Config& config);

    NetworkDescription::Ptr _network;
    Logger _log;

    InferenceManagerDemo _app;

    InferenceEngine::BlobMap _inputs;
};

}  // namespace IMD
}  // namespace vpux

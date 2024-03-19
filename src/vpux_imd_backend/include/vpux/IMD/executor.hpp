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
#include "vpux_private_properties.hpp"

#include <string>

namespace vpux {

class IMDExecutor final : public Executor {
public:
    struct InferenceManagerDemo;

    IMDExecutor(InferenceEngine::VPUXConfigParams::VPUXPlatform platform, const NetworkDescription::CPtr network,
                const Config& config);

    const NetworkDescription& getNetworkDesc() {
        return *_network.get();
    }

    InferenceManagerDemo& getApp() {
        return _app;
    }

    struct InferenceManagerDemo final {
        std::string elfFile;
        std::string runProgram;
        SmallVector<std::string> runArgs;
        int64_t timeoutSec;
        std::string chipsetArg;
        std::string imdElfArg;
    };

private:
    std::string getMoviToolsPath(const Config& config);
    std::string getSimicsPath(const Config& config);
    void setElfFile(const std::string& bin);
    void setMoviSimRunArgs(InferenceEngine::VPUXConfigParams::VPUXPlatform platform, const Config& config);
    void setMoviDebugRunArgs(InferenceEngine::VPUXConfigParams::VPUXPlatform platform, const Config& config);
    void setSimicsRunArgs(InferenceEngine::VPUXConfigParams::VPUXPlatform platform, const Config& config);

    static bool isValidElfSignature(StringRef filePath);
    void parseAppConfig(InferenceEngine::VPUXConfigParams::VPUXPlatform platform, const Config& config);

    NetworkDescription::CPtr _network;
    Logger _log;

    InferenceManagerDemo _app;
};

}  // namespace vpux

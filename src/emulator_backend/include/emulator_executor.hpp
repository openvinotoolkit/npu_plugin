//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"

#include <emu/manager.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ie = InferenceEngine;

namespace vpux {

class EmulatorExecutor final : public vpux::Executor {
public:
    EmulatorExecutor(const vpux::NetworkDescription::Ptr& network, const vpux::Config& config);

    void setup(const InferenceEngine::ParamMap&) final {
    }

    void push(const InferenceEngine::BlobMap& inputs) final;
    void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) final;

    void pull(InferenceEngine::BlobMap& outputs) final;

    bool isPreProcessingSupported(const PreprocMap&) const final {
        return false;
    }
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() final {
        return {};
    }
    InferenceEngine::Parameter getParameter(const std::string&) const final {
        return {};
    }

private:
    ie::Blob::Ptr repackTensor(const ie::Blob::Ptr&, const ie::TensorDesc&);

    Logger _logger;
    vpux::NetworkDescription::Ptr _network;
    mv::emu::Manager _manager;
};

}  // namespace vpux

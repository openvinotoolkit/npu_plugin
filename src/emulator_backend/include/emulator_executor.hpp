//
// Copyright 2019-2020 Intel Corporation.
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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <emu/manager.hpp>
#include <vpux.hpp>

namespace ie = InferenceEngine;

namespace vpux {

class EmulatorExecutor final : public vpux::Executor {
public:
    EmulatorExecutor(const vpux::NetworkDescription::Ptr& network);

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
    vpu::Logger _logger;
    vpux::NetworkDescription::Ptr _network;
    mv::emu::Manager _manager;
};

}  // namespace vpux

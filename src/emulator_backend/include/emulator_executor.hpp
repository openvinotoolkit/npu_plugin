//
// Copyright 2019-2020 Intel Corporation.
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

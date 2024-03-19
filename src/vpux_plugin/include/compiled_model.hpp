//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/runtime/icompiled_model.hpp>

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {

class CompiledModel final : public ov::ICompiledModel {
public:
    explicit CompiledModel(const std::shared_ptr<const ov::Model> model,
                           const std::shared_ptr<const ov::IPlugin> plugin,
                           const std::shared_ptr<const NetworkDescription> networkDescription,
                           const std::shared_ptr<Device> device, const Config& config);

    CompiledModel(const CompiledModel&) = delete;

    CompiledModel(CompiledModel&&) = delete;

    CompiledModel& operator=(const CompiledModel&) = delete;

    CompiledModel&& operator=(CompiledModel&&) = delete;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

private:
    void initialize_properties();

    void configure_stream_executors();

    const NetworkDescription::CPtr _networkPtr;
    const std::shared_ptr<const ov::Model> _model;
    const Config _config;
    Logger _logger;
    const Device::Ptr _device;
    mutable Executor::Ptr _executorPtr;
    std::shared_ptr<ov::threading::ITaskExecutor> _resultExecutor;

    // properties map: {name -> [supported, mutable, eval function]}
    std::map<std::string, std::tuple<bool, ov::PropertyMutability, std::function<ov::Any(const Config&)>>> _properties;
    std::vector<ov::PropertyName> _supportedProperties;
};

}  //  namespace vpux

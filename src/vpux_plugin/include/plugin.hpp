//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include <openvino/runtime/iplugin.hpp>

#include "vpux.hpp"
#include "vpux_backends.hpp"
#include "vpux_compiler.hpp"
#include "vpux_metrics.hpp"

#include "vpux/utils/IE/config.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {

class Plugin : public ov::IPlugin {
public:
    Plugin();

    Plugin(const Plugin&) = delete;

    Plugin(Plugin&&) = delete;

    Plugin& operator=(const Plugin&) = delete;

    Plugin&& operator=(Plugin&&) = delete;

    virtual ~Plugin() = default;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& modelPath,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& modelPath,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

private:
    VPUXBackends::Ptr _backends;
    mutable std::shared_ptr<Compiler> _compiler;

    std::map<std::string, std::string> _config;
    std::shared_ptr<OptionsDesc> _options;
    Config _globalConfig;
    Logger _logger;
    std::unique_ptr<Metrics> _metrics;

    // properties map: {name -> [supported, mutable, eval function]}
    std::map<std::string, std::tuple<bool, ov::PropertyMutability, std::function<ov::Any(const Config&)>>> _properties;
    std::vector<ov::PropertyName> _supportedProperties;

    static std::atomic<int> _compiledModelLoadCounter;
};

}  // namespace vpux

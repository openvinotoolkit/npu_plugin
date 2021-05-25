//
// Copyright 2019-2021 Intel Corporation.
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

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include <vpux/vpux_compiler_config.hpp>
#include <common_test_utils/test_common.hpp>

#include "kmb_test_utils.hpp"

using namespace InferenceEngine;

using BlobVector = std::vector<Blob::Ptr>;

struct PortInfo final {
    std::string layerName;
    size_t index = 0;

    PortInfo() = default;
    PortInfo(std::string layerName, size_t index = 0): layerName(std::move(layerName)), index(index) {}
};

class TestNetwork final {
public:
    using NodePtr = std::shared_ptr<ngraph::Node>;
    using Output = ngraph::Output<ngraph::Node>;
    using OutputVector = std::vector<Output>;
    using RefFunc = std::function<BlobVector(const NodePtr&, const BlobVector&, const TestNetwork&)>;

    TestNetwork() = default;

    TestNetwork(TestNetwork&& other) = default;
    TestNetwork& operator=(TestNetwork&& other) = default;

    TestNetwork(const TestNetwork& other);
    TestNetwork& operator=(const TestNetwork& other);

    void swap(TestNetwork& other);

    TestNetwork& addNetInput(
            const std::string& name,
            const SizeVector& dims,
            const Precision& precision);
    TestNetwork& setUserInput(
            const std::string& name,
            const Precision& precision,
            const Layout& layout);

    TestNetwork& addNetOutput(
            const PortInfo& port);
    TestNetwork& setUserOutput(
            const PortInfo& port,
            const Precision& precision,
            const Layout& layout);

    TestNetwork& addConst(const std::string& name, const std::shared_ptr<ngraph::op::Constant>& node);
    TestNetwork& addConst(const std::string& name, Blob::Ptr blob);
    TestNetwork& addConst(const std::string& name, float val) {
        return addConst(name, vpux::makeScalarBlob(val));
    }

    TestNetwork& addLayer(const std::string& name, const NodePtr& node, const RefFunc& refFunc);
    template <class LayerDef, typename... Args>
    LayerDef addLayer(Args&&... args) {
        return LayerDef(*this, std::forward<Args>(args)...);
    }

    TestNetwork& useCustomLayers(KernelType kernelType = KernelType::Native) {
        switch (kernelType) {
            case KernelType::Native:
                _compileConfig[VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS)] = "";
                break;
            case KernelType::Ocl:
                _compileConfig[VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS)] = _customOclLayerXmlDefault;
                break;
            case KernelType::Cpp:
                _compileConfig[VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS)] = _customCppLayerXmlDefault;
                break;
            default:
                throw std::invalid_argument("Undefined kernel type: " + ::testing::PrintToString(kernelType));
        }

        allowNCHWLayoutForMcmModelInput(true);
        return *this;
    }

    TestNetwork& useCustomLayers(const std::string& name) {
        _compileConfig[VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS)] = name;
        allowNCHWLayoutForMcmModelInput(true);
        return *this;
    }

    TestNetwork& useExtension(const std::string& name) {
        _exts.push_back(make_so_pointer<IExtension>(name));
        return *this;
    }

    const std::vector<IExtensionPtr>& getExtensions() const {
        return _exts;
    }

    TestNetwork& disableMcmPasses(const std::vector<std::pair<std::string, std::string>>& banList) {
        const auto passFold = [](std::string list, const std::pair<std::string, std::string>& pass) {
            return std::move(list) + pass.first + "," + pass.second + ";";
        };

        auto configValue = std::accumulate(begin(banList), end(banList), std::string{}, passFold);
        configValue.pop_back();

        _compileConfig[VPU_COMPILER_CONFIG_KEY(COMPILATION_PASS_BAN_LIST)] = std::move(configValue);
        return *this;
    }

    TestNetwork& allowNCHWLayoutForMcmModelInput(bool value = true) {
        _compileConfig[VPU_COMPILER_CONFIG_KEY(ALLOW_NCHW_MCM_INPUT)] =
            value ? InferenceEngine::PluginConfigParams::YES
                  : InferenceEngine::PluginConfigParams::NO;
        return *this;
    }

    void finalize(const std::string& netName = "") {
        _func = std::make_shared<ngraph::Function>(_results, _params, netName);
    }

    CNNNetwork getCNNNetwork() const;

    BlobMap calcRef(const BlobMap& inputs) const;

    NodePtr getLayer(const std::string& name) const { return _nodes.at(name); }
    Output getPort(const PortInfo& info) const { return Output(getLayer(info.layerName), info.index); }

    std::vector<DataPtr> getInputsInfo() const;
    std::vector<DataPtr> getOutputsInfo() const;

    const std::map<std::string, std::string>& compileConfig() const {
        return _compileConfig;
    }
    TestNetwork& setCompileConfig(const std::map<std::string, std::string>& compileConfig) {
        _compileConfig = compileConfig;
        return *this;
    }

private:
    std::unordered_map<std::string, NodePtr> _nodes;
    std::unordered_map<std::string, RefFunc> _refFuncs;

    std::unordered_map<std::string, Precision> _inputPrecisions;
    std::unordered_map<std::string, Layout> _inputLayouts;

    std::unordered_map<std::string, Precision> _outputPrecisions;
    std::unordered_map<std::string, Layout> _outputLayouts;

    ngraph::ParameterVector _params;
    ngraph::ResultVector _results;
    std::shared_ptr<ngraph::Function> _func;

    std::map<std::string, std::string> _compileConfig;

    static const std::string _customOclLayerXmlDefault;
    static const std::string _customCppLayerXmlDefault;

    std::vector<IExtensionPtr> _exts;
};

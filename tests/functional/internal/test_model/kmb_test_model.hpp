//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <common_test_utils/test_common.hpp>
#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include <vpux/vpux_compiler_config.hpp>

#include "kmb_test_utils.hpp"

using namespace InferenceEngine;

using BlobVector = std::vector<Blob::Ptr>;

struct PortInfo final {
    std::string layerName;
    size_t index = 0;

    PortInfo() = default;
    PortInfo(std::string layerName, size_t index = 0): layerName(std::move(layerName)), index(index) {
    }
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

    TestNetwork& addNetInput(const std::string& name, const SizeVector& dims, const Precision& precision);
    TestNetwork& setUserInput(const std::string& name, const Precision& precision, const Layout& layout);

    TestNetwork& addNetOutput(const PortInfo& port);
    TestNetwork& setUserOutput(const PortInfo& port, const Precision& precision, const Layout& layout);

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

    TestNetwork& useExtension(const std::string& name) {
        _exts.push_back(std::make_shared<Extension>(name));
        return *this;
    }

    const std::vector<IExtensionPtr>& getExtensions() const {
        return _exts;
    }

    void finalize(const std::string& netName = "") {
        _func = std::make_shared<ngraph::Function>(_results, _params, netName);
    }

    CNNNetwork getCNNNetwork() const;

    BlobMap calcRef(const BlobMap& inputs) const;

    NodePtr getLayer(const std::string& name) const {
        return _nodes.at(name);
    }
    Output getPort(const PortInfo& info) const {
        return Output(getLayer(info.layerName), info.index);
    }

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

    std::vector<IExtensionPtr> _exts;
};

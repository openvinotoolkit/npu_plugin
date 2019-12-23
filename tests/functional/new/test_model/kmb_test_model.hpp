//
// Copyright 2019 Intel Corporation.
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

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>

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
        return addConst(name, makeScalarBlob(val));
    }

    TestNetwork& addLayer(const std::string& name, const NodePtr& node, const RefFunc& refFunc);
    template <class LayerDef, typename... Args>
    LayerDef addLayer(Args&&... args) {
        return LayerDef(*this, std::forward<Args>(args)...);
    }

    void finalize() {
        _func = std::make_shared<ngraph::Function>(_results, _params);
    }

    CNNNetwork toCNNNetwork() const;

    BlobMap calcRef(const BlobMap& inputs) const;

    NodePtr getLayer(const std::string& name) const { return _nodes.at(name); }
    Output getPort(const PortInfo& info) const { return Output(getLayer(info.layerName), info.index); }

    std::vector<DataPtr> getInputsInfo() const;
    std::vector<DataPtr> getOutputsInfo() const;

    const std::map<std::string, std::string>& compileConfig() const {
        return _compileConfig;
    }
    void setCompileConfig(const std::map<std::string, std::string>& compileConfig) {
        _compileConfig = compileConfig;
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
};

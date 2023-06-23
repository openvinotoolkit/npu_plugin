//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ovhelper.hpp"

#include <inference_engine.hpp>
// FIXME: It might be incompatible with
// old OpenVINO without 2.0 API.
#include <openvino/openvino.hpp>

#include <opencv2/gapi/infer/ie.hpp>

#include <fstream>

namespace IE = InferenceEngine;

static int toPrecision(IE::Precision prec) {
    switch (prec) {
    case IE::Precision::U8:
        return CV_8U;
    case IE::Precision::FP32:
        return CV_32F;
    case IE::Precision::FP16:
        return CV_16F;
    }
    throw std::logic_error("Unsupported IE precision");
}

static int toPrecision(ov::element::Type prec) {
    switch (prec) {
    case ov::element::u8:
        return CV_8U;
    case ov::element::f32:
        return CV_32F;
    case ov::element::f16:
        return CV_16F;
    }
    throw std::logic_error("Unsupported OV precision");
}

static std::vector<int> toDims(const std::vector<size_t>& sz_vec) {
    std::vector<int> result;
    result.reserve(sz_vec.size());
    for (auto sz : sz_vec) {
        // FIXME: Probably requires some check...
        result.push_back(static_cast<int>(sz));
    }
    return result;
}

class IELayersReader : public ILayersReader {
public:
    InOutLayers readFromModel(const std::string& model, const std::string& bin) override;

    InOutLayers readFromBlob(const std::string& blob, const std::string& device) override;

private:
    InferenceEngine::Core m_core;
};

template <typename InfoMap>
std::vector<LayerInfo> ieToLayersInfo(const InfoMap& info) {
    std::vector<LayerInfo> layers;
    layers.reserve(info.size());
    std::transform(info.begin(), info.end(), std::back_inserter(layers), [](const auto& it) {
        const auto& desc = it.second->getTensorDesc();
        return LayerInfo{it.first, toDims(desc.getDims()), toPrecision(desc.getPrecision())};
    });
    return layers;
};

template <typename InfoVec>
std::vector<LayerInfo> ovToLayersInfo(const InfoVec& vec) {
    std::vector<LayerInfo> layers;
    layers.reserve(vec.size());
    std::transform(vec.begin(), vec.end(), std::back_inserter(layers), [](const auto& node) {
        return LayerInfo{node.get_any_name(), toDims(node.get_shape()), toPrecision(node.get_element_type())};
    });
    return layers;
};

InOutLayers IELayersReader::readFromModel(const std::string& xml, const std::string& bin) {
    IE::CNNNetwork network = m_core.ReadNetwork(xml, bin);

    auto input_layers = ieToLayersInfo(network.getInputsInfo());
    auto output_layers = ieToLayersInfo(network.getOutputsInfo());

    return {std::move(input_layers), std::move(output_layers)};
}

InOutLayers IELayersReader::readFromBlob(const std::string& blob, const std::string& device) {
    IE::ExecutableNetwork exec = m_core.ImportNetwork(blob, device);

    auto input_layers = ieToLayersInfo(exec.GetInputsInfo());
    auto output_layers = ieToLayersInfo(exec.GetOutputsInfo());

    return {std::move(input_layers), std::move(output_layers)};
}

class OVLayersReader : public ILayersReader {
public:
    InOutLayers readFromModel(const std::string& xml, const std::string& bin) override;

    InOutLayers readFromBlob(const std::string& blob, const std::string& device) override;

private:
    ov::Core m_core;
};

InOutLayers OVLayersReader::readFromModel(const std::string& xml, const std::string& bin) {
    auto model = m_core.read_model(xml, bin);

    auto input_layers = ovToLayersInfo(model->inputs());
    auto output_layers = ovToLayersInfo(model->outputs());

    return {std::move(input_layers), std::move(output_layers)};
}

InOutLayers OVLayersReader::readFromBlob(const std::string& blob, const std::string& device) {
    std::ifstream file(blob, std::ios_base::in | std::ios_base::binary);
    GAPI_Assert(file.is_open());
    auto compiled_model = m_core.import_model(file, device);

    auto input_layers = ovToLayersInfo(compiled_model.inputs());
    auto output_layers = ovToLayersInfo(compiled_model.outputs());

    return {std::move(input_layers), std::move(output_layers)};
}

ILayersReader::Ptr ILayersReader::create(const bool use_ov_old_api) {
    if (use_ov_old_api) {
        return std::make_shared<IELayersReader>();
    }
    return std::make_shared<OVLayersReader>();
}

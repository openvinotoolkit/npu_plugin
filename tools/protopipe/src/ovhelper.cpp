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

std::vector<std::string> extractLayerNames(const std::vector<LayerInfo>& layers) {
    std::vector<std::string> names;
    std::transform(layers.begin(), layers.end(), std::back_inserter(names), [](const auto& layer) {
        return layer.name;
    });
    return names;
}

static ov::element::Type toElementType(int cvdepth) {
    switch (cvdepth) {
    case CV_8U:
        return ov::element::u8;
    case CV_32F:
        return ov::element::f32;
    case CV_16F:
        return ov::element::f16;
    }
    throw std::logic_error("Failed to convert opencv depth to ov::element::Type");
}

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

static void alignDimsWithLayout(std::vector<int>& ie_dims, const IE::Layout layout) {
    switch (layout) {
    case IE::Layout::NDHWC:
        ie_dims = {ie_dims[0], ie_dims[2], ie_dims[3], ie_dims[4], ie_dims[1]};
        break;
    case IE::Layout::NHWC:
        ie_dims = {ie_dims[0], ie_dims[2], ie_dims[3], ie_dims[1]};
        break;
    case IE::Layout::HWC:
        ie_dims = {ie_dims[1], ie_dims[2], ie_dims[0]};
        break;
    }
}

static IE::Layout toLayout(const std::string& layout) {
    const std::map<std::string, IE::Layout> mapping = {
            {"NCDHW", IE::Layout::NCDHW}, {"NDHWC", IE::Layout::NDHWC}, {"NCHW", IE::Layout::NCHW},
            {"NHWC", IE::Layout::NHWC},   {"CHW", IE::Layout::CHW},     {"HWC", IE::Layout::HWC},
            {"NC", IE::Layout::NC},       {"HW", IE::Layout::HW},       {"C", IE::Layout::C},
    };
    const auto it = mapping.find(layout);
    if (it == mapping.end()) {
        throw std::logic_error("Unsupported IE layout: " + layout);
    }
    return it->second;
}

static void cfgInputPreproc(ov::preprocess::PrePostProcessor& ppp, const std::shared_ptr<ov::Model>& model,
                            const AttrMap<int>& input_precision, const AttrMap<std::string>& input_layout,
                            const AttrMap<std::string>& input_model_layout) {
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        auto& ii = ppp.input(name);

        const auto ip = lookUp(input_precision, name);
        if (ip.has_value()) {
            ii.tensor().set_element_type(toElementType(*ip));
        }

        const auto il = lookUp(input_layout, name);
        if (il.has_value()) {
            ii.tensor().set_layout(ov::Layout(*il));
        }

        const auto iml = lookUp(input_model_layout, name);
        if (iml.has_value()) {
            ii.model().set_layout(ov::Layout(*iml));
        }
    }
}

static void cfgOutputPostproc(ov::preprocess::PrePostProcessor& ppp, const std::shared_ptr<ov::Model>& model,
                              const AttrMap<int>& output_precision, const AttrMap<std::string>& output_layout,
                              const AttrMap<std::string> output_model_layout) {
    for (const auto& output : model->outputs()) {
        const auto& name = output.get_any_name();
        auto& oi = ppp.output(name);

        const auto op = lookUp(output_precision, name);
        if (op.has_value()) {
            oi.tensor().set_element_type(toElementType(*op));
        }

        const auto ol = lookUp(output_layout, name);
        if (ol.has_value()) {
            oi.tensor().set_layout(ov::Layout(*ol));
        }

        const auto oml = lookUp(output_model_layout, name);
        if (oml.has_value()) {
            oi.model().set_layout(ov::Layout(*oml));
        }
    }
}

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

static void cfgIELayersInfo(std::vector<LayerInfo>& layers, AttrMap<int> precisions, AttrMap<std::string> layouts) {
    for (auto& layer : layers) {
        const auto prec = lookUp(precisions, layer.name);
        if (prec.has_value()) {
            layer.prec = *prec;
        }
        const auto layout = lookUp(layouts, layer.name);
        if (layout.has_value()) {
            alignDimsWithLayout(layer.dims, toLayout(*layout));
        }
    }
}

class IELayersReader : public ILayersReader {
public:
    InOutLayers readFromModel(const std::string& model, const std::string& bin,
                              const PrePostProccesingInfo& info) override;

    InOutLayers readFromBlob(const std::string& blob, const std::string& device,
                             const std::map<std::string, std::string>& config) override;

private:
    InferenceEngine::Core m_core;
};

InOutLayers IELayersReader::readFromModel(const std::string& xml, const std::string& bin,
                                          const PrePostProccesingInfo& info) {
    IE::CNNNetwork network = m_core.ReadNetwork(xml, bin);

    auto input_layers = ieToLayersInfo(network.getInputsInfo());
    const auto input_names = extractLayerNames(input_layers);
    const auto input_precision = unpackLayerAttr(info.input_precision, input_names, "input precision");
    const auto input_layout = unpackLayerAttr(info.input_layout, input_names, "input layout");
    cfgIELayersInfo(input_layers, input_precision, input_layout);

    auto output_layers = ieToLayersInfo(network.getOutputsInfo());
    const auto output_names = extractLayerNames(output_layers);
    const auto output_precision = unpackLayerAttr(info.output_precision, input_names, "output precision");
    const auto output_layout = unpackLayerAttr(info.output_layout, input_names, "output layout");
    cfgIELayersInfo(output_layers, output_precision, output_layout);

    return {std::move(input_layers), std::move(output_layers)};
}

InOutLayers IELayersReader::readFromBlob(const std::string& blob, const std::string& device,
                                         const std::map<std::string, std::string>& config) {
    IE::ExecutableNetwork exec = m_core.ImportNetwork(blob, device, config);

    const auto& inputs_info = exec.GetInputsInfo();
    auto input_layers = ieToLayersInfo(inputs_info);
    for (auto& layer : input_layers) {
        const auto layout = inputs_info.at(layer.name)->getTensorDesc().getLayout();
        alignDimsWithLayout(layer.dims, layout);
    }

    const auto& outputs_info = exec.GetOutputsInfo();
    auto output_layers = ieToLayersInfo(outputs_info);
    for (auto& layer : output_layers) {
        const auto layout = outputs_info.at(layer.name)->getTensorDesc().getLayout();
        alignDimsWithLayout(layer.dims, layout);
    }

    return {std::move(input_layers), std::move(output_layers)};
}

class OVLayersReader : public ILayersReader {
public:
    InOutLayers readFromModel(const std::string& xml, const std::string& bin,
                              const PrePostProccesingInfo& info) override;

    InOutLayers readFromBlob(const std::string& blob, const std::string& device,
                             const std::map<std::string, std::string>& config) override;

private:
    ov::Core m_core;
};

static std::vector<std::string> extractLayerNames(const std::vector<ov::Output<ov::Node>>& nodes) {
    std::vector<std::string> names;
    std::transform(nodes.begin(), nodes.end(), std::back_inserter(names), [](const auto& node) {
        return node.get_any_name();
    });
    return names;
}

InOutLayers OVLayersReader::readFromModel(const std::string& xml, const std::string& bin,
                                          const PrePostProccesingInfo& info) {
    auto model = m_core.read_model(xml, bin);
    {
        ov::preprocess::PrePostProcessor ppp(model);

        const auto& input_names = extractLayerNames(model->inputs());
        const auto ip_map = unpackLayerAttr(info.input_precision, input_names, "input precision");
        const auto il_map = unpackLayerAttr(info.input_layout, input_names, "input layout");
        const auto iml_map = unpackLayerAttr(info.input_model_layout, input_names, "input model layout");
        cfgInputPreproc(ppp, model, ip_map, il_map, iml_map);

        const auto& output_names = extractLayerNames(model->outputs());
        const auto op_map = unpackLayerAttr(info.output_precision, output_names, "output precision");
        const auto ol_map = unpackLayerAttr(info.output_layout, output_names, "output layout");
        const auto oml_map = unpackLayerAttr(info.output_model_layout, output_names, "output model layout");
        cfgOutputPostproc(ppp, model, op_map, ol_map, oml_map);

        model = ppp.build();
    }

    auto input_layers = ovToLayersInfo(model->inputs());
    auto output_layers = ovToLayersInfo(model->outputs());

    return {std::move(input_layers), std::move(output_layers)};
}

InOutLayers OVLayersReader::readFromBlob(const std::string& blob, const std::string& device,
                                         const std::map<std::string, std::string>& config) {
    std::ifstream file(blob, std::ios_base::in | std::ios_base::binary);
    GAPI_Assert(file.is_open());
    auto compiled_model = m_core.import_model(file, device, {config.begin(), config.end()});

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

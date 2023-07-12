//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/util/variant.hpp>

#include "utils.hpp"

// NB: model might be *.xml & *.bin, *.onnx, *.pdpd
struct ModelPath {
    std::string model;
    std::string bin;
};
struct BlobPath {
    std::string path;
};

using Path = cv::util::variant<ModelPath, BlobPath>;

template <typename T>
using AttrMap = std::map<std::string, T>;
// NB: This type is supposed to be used to hold in/out layers
// attributes such as precision, layout, shape etc.
//
// User can provide attributes either:
// 1. cv::util::monostate - No value specified explicitly.
// 2. Attr - value specified explicitly that should be broadcasted to all layers.
// 3. AttrMap[str->T] - map specifies value for particular layer.
template <typename Attr>
using LayerVariantAttr = cv::util::variant<cv::util::monostate, AttrMap<Attr>, Attr>;

struct LayerInfo {
    std::string name;
    std::vector<int> dims;
    int prec;
};

std::vector<std::string> extractLayerNames(const std::vector<LayerInfo>& layers);

struct InOutLayers {
    std::vector<LayerInfo> in_layers;
    std::vector<LayerInfo> out_layers;
};

struct PrePostProccesingInfo {
    LayerVariantAttr<int> input_precision;
    LayerVariantAttr<int> output_precision;
    LayerVariantAttr<std::string> input_layout;
    LayerVariantAttr<std::string> output_layout;
    LayerVariantAttr<std::string> input_model_layout;
    LayerVariantAttr<std::string> output_model_layout;
};

template <typename K, typename V>
cv::optional<V> lookUp(const std::map<K, V>& map, const K& key) {
    const auto it = map.find(key);
    if (it == map.end()) {
        return {};
    }
    return cv::util::make_optional(std::move(it->second));
}

template <typename T>
static AttrMap<T> unpackLayerAttr(const LayerVariantAttr<T>& attr, const std::vector<std::string>& layer_names,
                                  const std::string& attrname) {
    AttrMap<T> attrmap;
    if (cv::util::holds_alternative<T>(attr)) {
        auto value = cv::util::get<T>(attr);
        for (const auto& name : layer_names) {
            attrmap.emplace(name, value);
        }
    } else if (cv::util::holds_alternative<AttrMap<T>>(attr)) {
        attrmap = cv::util::get<AttrMap<T>>(attr);
        std::unordered_set<std::string> layers_set{layer_names.begin(), layer_names.end()};
        for (const auto& [name, attr] : attrmap) {
            const auto it = layers_set.find(name);
            if (it == layers_set.end()) {
                throw std::logic_error("Failed to find layer \"" + name + "\" to specify " + attrname);
            }
        }
    }
    return attrmap;
}

struct ILayersReader {
    using Ptr = std::shared_ptr<ILayersReader>;

    static Ptr create(const bool use_ov_old_api);

    virtual InOutLayers readFromModel(const std::string& model, const std::string& bin,
                                      const PrePostProccesingInfo& info) = 0;

    virtual InOutLayers readFromBlob(const std::string& blob, const std::string& device,
                                     const std::map<std::string, std::string>& config) = 0;
};

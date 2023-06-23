//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>

#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/util/variant.hpp>

#include "utils.hpp"

struct LayerInfo {
    std::string name;
    std::vector<int> dims;
    int prec;
};

struct InOutLayers {
    std::vector<LayerInfo> in_layers;
    std::vector<LayerInfo> out_layers;
};

struct ILayersReader {
    using Ptr = std::shared_ptr<ILayersReader>;

    static Ptr create(const bool use_ov_old_api);

    virtual InOutLayers readFromModel(const std::string& model, const std::string& bin) = 0;

    virtual InOutLayers readFromBlob(const std::string& blob, const std::string& device) = 0;
};

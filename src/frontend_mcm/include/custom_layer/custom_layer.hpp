// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>

#include <details/caseless.hpp>
#include <functional>
#include <custom_layer/custom_kernel.hpp>
#include <map>
#include <memory>
#include <pugixml.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/small_vector.hpp>

namespace vpu {

namespace ie = InferenceEngine;

class CustomLayer final {
public:
    using Ptr = std::shared_ptr<CustomLayer>;
    explicit CustomLayer(std::string configDir, const pugi::xml_node& customLayer);

    std::vector<CustomKernel> kernels() const { return _kernels; }
    std::string layerName() const { return _layerName; }
    std::map<int, ie::Layout> inputs() { return _inputs; }
    std::map<int, ie::Layout> outputs() { return _outputs; }

    static ie::details::caseless_map<std::string, std::vector<CustomLayer::Ptr>> loadFromFile(
                const std::string& configFile,
                bool canBeMissed = false);

    bool meetsWhereRestrictions(const std::map<std::string, std::string>& params) const;
    static bool isLegalSizeRule(const std::string& rule, std::map<std::string, std::string> layerParams);
    static InferenceEngine::Layout formatToLayout(const CustomDataFormat& format);
private:
    std::string _configDir;
    std::string _layerName;
    std::map<std::string, std::string> _whereParams;

    std::vector<CustomKernel> _kernels;

    std::map<int, ie::Layout> _inputs;
    std::map<int, ie::Layout> _outputs;
};

};  // namespace vpu

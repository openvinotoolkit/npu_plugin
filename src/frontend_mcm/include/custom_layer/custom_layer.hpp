// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>

#include <caseless.hpp>
#include <custom_layer/custom_kernel.hpp>
#include <functional>
#include <map>
#include <memory>
#include <pugixml.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/small_vector.hpp>
#include <vpu/utils/logger.hpp>
#include <include/mcm/op_model.hpp>

namespace vpu {

namespace ie = InferenceEngine;

class CustomLayer final {
public:
    using Ptr = std::shared_ptr<CustomLayer>;
    explicit CustomLayer(std::string configDir, const pugi::xml_node& customLayer);

    std::vector<CustomKernel::Ptr> kernels() const {
        return _kernels;
    }
    std::string layerName() const {
        return _layerName;
    }
    std::map<int, ie::Layout> inputs() const {
        return _inputs;
    }
    std::map<int, ie::Layout> outputs() const {
        return _outputs;
    }

    static ie::details::caseless_map<std::string, std::vector<CustomLayer::Ptr>> loadFromFile(
            const std::string& configFile, bool canBeMissed = false);

    bool meetsWhereRestrictions(const std::map<std::string, std::string>& params) const;
    static bool isLegalSizeRule(const std::string& rule, std::map<std::string, std::string> layerParams);
    static InferenceEngine::Layout formatToLayout(const CustomDataFormat& format);

private:
    std::string _configDir;
    std::string _layerName;
    std::unordered_map<std::string, std::string> _whereParams;

    std::vector<CustomKernel::Ptr> _kernels;

    std::map<int, ie::Layout> _inputs;
    std::map<int, ie::Layout> _outputs;
};

class SizeRuleValidator : public CustomKernelVisitor {
public:
    explicit SizeRuleValidator(CustomLayer::Ptr customLayer,
                               const std::map<std::string, std::string>& cnnLayerParams,
                               Logger::Ptr logger = {});

    void visitCpp(const CustomKernelCpp& kernel) override;
    void visitCL(const CustomKernelOcl& kernel) override;

    bool result() const { return _result; }

private:
    CustomLayer::Ptr _customLayer;
    const std::map<std::string, std::string>& _cnnLayerParams;
    Logger::Ptr _logger;
    bool _result = false;
};

class OperationFactory : public CustomKernelVisitor {
public:
    explicit OperationFactory(int stageIdx, mv::OpModel& modelMcm,
                              const std::vector<uint8_t>& kernelData,
                              const std::vector<mv::Data::TensorIterator>& stageInputs,
                              const std::vector<mv::TensorInfo>& stageOutputs,
                              const std::string& friendlyName);

    void visitCpp(const CustomKernelCpp& kernel) override;
    void visitCL(const CustomKernelOcl& kernel) override;

    mv::Data::TensorIterator result() const { return _result; }

private:
    int _stageIdx = 0;
    mv::OpModel& _modelMcm;
    const std::vector<uint8_t>& _kernelData;
    const std::vector<mv::Data::TensorIterator>& _stageInputs;
    const std::vector<mv::TensorInfo>& _stageOutputs;
    const std::string& _friendlyName;
    mv::Data::TensorIterator _result;
};

};  // namespace vpu

// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pugixml.hpp>
#include <ie_common.h>

#include <vpu/utils/enums.hpp>
#include <vpu/utils/small_vector.hpp>

namespace vpu {

namespace ie = InferenceEngine;

VPU_DECLARE_ENUM(CustomParamType,
    Input,
    Output,
    Data,
    LocalData,
    InputBuffer,
    OutputBuffer,
    Int,
    Float)

VPU_DECLARE_ENUM(CustomDataFormat,
                 BYXF = 0,  // NHWC used in most software layers
                 BFYX = 1,  // NCHW used if HW module is enabled
                 YXF = 2,   // HWC used in most software layers
                 FYX = 3,   // CHW used if HW module is enabled
                 BF = 4,    // NC layout
                 Any = 5)   // doesn't really matter

VPU_DECLARE_ENUM(CustomDimSource, Input, Output)

struct CustomKernel final {
    struct BindingParameter final {
        CustomParamType type = CustomParamType::Input;
        CustomDataFormat format = CustomDataFormat::Any;
        std::string argName;
        int portIndex = -1;
        std::string irSource;
        std::string bufferSizeRule;
        CustomDimSource dimSource;
        int dimIdx = -1;
    };

    struct Argument {
        std::string name;
        int underlyingTypeSize{};

        Argument(std::string name, int typeSize): name{std::move(name)}, underlyingTypeSize{typeSize} {}
    };

private:
    std::string _configDir;
    int _maxShaves = 0;
    std::vector<uint8_t> _kernelBinary;
    std::unordered_map<std::string, BindingParameter> _bindings;
    SmallVector<std::string> _globalGridSizeRules;
    SmallVector<std::string> _localGridSizeRules;
    SmallVector<Argument> _kernelArguments;
    int _kernelId = 0;

    CustomDimSource _wgDimSource = CustomDimSource::Input;
    int _wgDimIdx = -1;

    int _inputDataCount = 0;

public:
    explicit CustomKernel(const pugi::xml_node& node, std::string configDir);

    void processParametersNode(const pugi::xml_node& node);
    void processWorkSizesNode(const pugi::xml_node& node);

    int maxShaves() const { return _maxShaves; }
    const std::vector<uint8_t>& kernelBinary() const { return _kernelBinary; }
    const SmallVector<Argument>& arguments() const { return _kernelArguments; }
    const std::unordered_map<std::string, BindingParameter>& bindings() const { return _bindings; }
    const SmallVector<std::string>& globalGridSizeRules() const { return _globalGridSizeRules; }
    const SmallVector<std::string>& localGridSizeRules() const { return _localGridSizeRules; }
    int kernelId() const { return _kernelId; }
    CustomDimSource dimSource() const { return _wgDimSource; }
    int dimSourceIndex() const { return _wgDimIdx; }
    int inputDataCount() const { return _inputDataCount; }
};

} // namespace vpu

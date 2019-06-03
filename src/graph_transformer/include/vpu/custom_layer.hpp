//
// Copyright 2016-2018 Intel Corporation.
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

#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <functional>

#include <details/caseless.hpp>

#include <pugixml.hpp>

#include <vpu/utils/enums.hpp>
#include <vpu/utils/containers.hpp>

namespace vpu {

namespace ie = InferenceEngine;

VPU_DECLARE_ENUM(CustomDataFormat,
    BYXF = 0,  // HWC used in most software layers
    BFYX = 1,  // CHW used if HW module is enabled
    Any  = 2,  // doesn't really matter
    None = 3
)

VPU_DECLARE_ENUM(CustomParamType,
    Input,
    Output,
    Data,
    InputBuffer,
    OutputBuffer,
    Int,
    Float
)

VPU_DECLARE_ENUM(CustomDimSource,
    Input,
    Output
)

class CustomLayer final {
public:
    using Ptr = std::shared_ptr<CustomLayer>;

    struct KernelParam final {
        CustomParamType type = CustomParamType::Input;
        CustomDataFormat format = CustomDataFormat::Any;
        std::string argName;
        int portIndex = -1;
        std::string irSource;
        SmallVector<std::string> bufferSizeRules;
        CustomDimSource dimSource;
        int dimIdx = -1;
    };

    static ie::details::caseless_map<std::string, CustomLayer::Ptr> loadFromFile(
                const std::string& configFile,
                bool canBeMissed = false);

    const std::string& kernelBinary() const { return _kernelBinary; }

    void setStageNumInputs(int id);
    int stageNumInputs() const;
    int kernelAddress(int idx = 1) const;
    int maxShaves() const;

    const SmallVector<KernelParam>& bindings() const { return _kernelParams; }
    const SmallVector<std::string>& parameters() const { return _parameters; }

    const SmallVector<std::string>& globalSizeRules() const { return _globalSizeRules; }
    const SmallVector<std::string>& localSizeRules() const { return _localSizeRules; }

    CustomDimSource dimSource() const { return _wgDimSource; }
    int dimSourceIndex() const { return _wgDimIdx; }

private:
    explicit CustomLayer(const std::string& dirname) : _configDir(dirname) {}

    void loadSingleLayer(const pugi::xml_node& node);
    void processKernelNode(const pugi::xml_node& node);
    void processParametersNode(const pugi::xml_node& node);
    void processWorkSizesNode(const pugi::xml_node& node);

    static bool isLegalSizeRule(const std::string& rule);
    static CustomDataFormat formatFromString(const std::string& str);

private:
    std::string _configDir;
    std::string _layerName;
    std::string _kernelEntry;
    std::string _kernelBinary;

    int _maxShaves = 0;
    int _stageNumInputs = -1;

    SmallVector<KernelParam> _kernelParams;
    SmallVector<std::string> _globalSizeRules;
    SmallVector<std::string> _localSizeRules;
    SmallVector<std::string> _parameters;

    std::map<uint32_t, uint32_t, std::greater<uint32_t>> _kernelAddress;

    CustomDimSource _wgDimSource = CustomDimSource::Input;
    int _wgDimIdx = -1;
};

};  // namespace vpu

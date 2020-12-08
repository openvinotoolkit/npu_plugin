// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <xml_parse_utils.h>

#include <caseless.hpp>
#include <custom_layer/custom_kernel.hpp>

namespace vpu {

std::vector<uint8_t> CustomKernel::loadKernelBinary(const pugi::xml_node& node, const std::string& configDir) {
    auto source = node.child("Source");
    if (source.empty()) {
        THROW_IE_EXCEPTION << "Kernel binary not found";
    }

    auto fileName = configDir + "/" + XMLParseUtils::GetStrAttr(source, "filename", "");
    std::ifstream inputFile(fileName, std::ios::in | std::ios::binary);
    if (!inputFile.is_open()) {
        THROW_IE_EXCEPTION << "Couldn't open kernel file " << fileName;
    }

    std::ostringstream contentStream;
    contentStream << inputFile.rdbuf();
    const auto& kernelData = contentStream.str();

    return std::vector<uint8_t>(begin(kernelData), end(kernelData));
}

std::pair<CustomDimSource, int> CustomKernel::parseDimSource(const std::string& dims) {
    const auto cmp = ie::details::CaselessEq<std::string>{};
    const auto pos = dims.find_first_of(',');
    const auto source = dims.substr(0, pos);
    const auto dimSource = [&] {
        if (cmp(source, "input")) {
            return CustomDimSource::Input;
        } else if (cmp(source, "output")) {
            return CustomDimSource::Output;
        } else {
            THROW_IE_EXCEPTION << "Invalid dim source argument" << source;
        }
    }();

    const auto idx = [&] {
        if (pos == std::string::npos) {
            return -1;
        }
        const auto idxString = dims.substr(pos + 1, std::string::npos);
        return std::stoi(idxString);
    }();

    return std::make_pair(dimSource, idx);
}

CustomDataFormat CustomKernel::formatFromString(const std::string& str) {
    static const ie::details::caseless_map<std::string, CustomDataFormat> FormatNameToType = {
            {"BFYX", CustomDataFormat::BFYX}, {"BYXF", CustomDataFormat::BYXF}, {"FYX", CustomDataFormat::FYX},
            {"YXF", CustomDataFormat::YXF},   {"BF", CustomDataFormat::BF},     {"ANY", CustomDataFormat::Any}};

    auto it = FormatNameToType.find(str);
    if (it != FormatNameToType.end()) {
        return it->second;
    }

    THROW_IE_EXCEPTION << "Tensor node has an invalid format '" << str << "'";
}

SmallVector<std::string> CustomKernel::parseSizeRule(const std::string& size) {
    auto result = SmallVector<std::string>();
    result.reserve(std::count(begin(size), end(size), ',') + 1);
    std::stringstream sizeRules{size};
    std::string bufferSize;

    while (std::getline(sizeRules, bufferSize, ',')) {
        result.push_back(bufferSize);
    }

    return result;
}

SmallVector<CustomKernel::BindingParameter> CustomKernel::processParametersNode(const pugi::xml_node& node) {
    const auto cmp = ie::details::CaselessEq<std::string>{};
    const auto parameters = node.child("Parameters");
    auto bindings = SmallVector<BindingParameter>{};

    for (auto tensor = parameters.child("Tensor"); !tensor.empty(); tensor = tensor.next_sibling("Tensor")) {
        BindingParameter kp;

        auto typeStr = XMLParseUtils::GetStrAttr(tensor, "type");
        if (cmp(typeStr, "input")) {
            kp.type = CustomParamType::Input;
        } else if (cmp(typeStr, "output")) {
            kp.type = CustomParamType::Output;
        } else if (cmp(typeStr, "input_buffer")) {
            kp.type = CustomParamType::InputBuffer;
        } else if (cmp(typeStr, "output_buffer")) {
            kp.type = CustomParamType::OutputBuffer;
        } else if (cmp(typeStr, "data")) {
            kp.type = CustomParamType::Data;
        } else {
            THROW_IE_EXCEPTION << "Tensor node has an invalid type '" << typeStr << "'";
        }

        if (kp.type == CustomParamType::InputBuffer || kp.type == CustomParamType::OutputBuffer) {
            const auto sizeRule = XMLParseUtils::GetStrAttr(tensor, "size");
            kp.bufferSizeRule = parseSizeRule(sizeRule)[0];

            const auto dimString = XMLParseUtils::GetStrAttr(tensor, "dim");
            std::tie(kp.dimSource, kp.dimIdx) = parseDimSource(dimString);
        }

        kp.format = formatFromString(XMLParseUtils::GetStrAttr(tensor, "format", "BFYX"));
        kp.argName = XMLParseUtils::GetStrAttr(tensor, "arg-name");
        kp.portIndex = XMLParseUtils::GetIntAttr(tensor, "port-index");

        bindings.push_back(std::move(kp));
    }

    for (auto data = parameters.child("Data"); !data.empty(); data = data.next_sibling("Data")) {
        BindingParameter kp;

        auto typeStr = XMLParseUtils::GetStrAttr(data, "type");
        if (cmp(typeStr, "data")) {
            kp.type = CustomParamType::Data;
        } else if (cmp(typeStr, "local_data")) {
            kp.type = CustomParamType::LocalData;
        } else {
            THROW_IE_EXCEPTION << "Data node has an invalid type '" << typeStr << "'";
        }

        kp.argName = XMLParseUtils::GetStrAttr(data, "arg-name");

        kp.irSource = XMLParseUtils::GetStrAttr(data, "source", "");
        const auto dimString = XMLParseUtils::GetStrAttr(data, "dim", "");

        if (kp.irSource.empty() && dimString.empty()) {
            THROW_IE_EXCEPTION << "Data node has no source or dim";
        }

        if (!kp.irSource.empty() && !dimString.empty()) {
            THROW_IE_EXCEPTION << "Data node can only have source or dim";
        }

        if (kp.type == CustomParamType::LocalData) {
            const auto bufferSize = XMLParseUtils::GetStrAttr(data, "size", "");
            kp.bufferSizeRule = bufferSize;

            if (!dimString.empty()) {
                std::tie(kp.dimSource, kp.dimIdx) = parseDimSource(dimString);
            }
        }

        bindings.push_back(std::move(kp));
    }

    for (auto scalar = parameters.child("Scalar"); !scalar.empty(); scalar = scalar.next_sibling("Scalar")) {
        BindingParameter kp;

        const auto type = XMLParseUtils::GetStrAttr(scalar, "type");
        if (cmp(type, "int")) {
            kp.type = CustomParamType::Int;
        } else if (cmp(type, "float")) {
            kp.type = CustomParamType::Float;
        } else {
            THROW_IE_EXCEPTION << "Scalar node has an invalid type " << type;
        }

        kp.argName = XMLParseUtils::GetStrAttr(scalar, "arg-name");
        kp.portIndex = XMLParseUtils::GetIntAttr(scalar, "port-index", -1);
        kp.irSource = XMLParseUtils::GetStrAttr(scalar, "source", "");

        bindings.push_back(std::move(kp));
    }

    return bindings;
}

}  // namespace vpu

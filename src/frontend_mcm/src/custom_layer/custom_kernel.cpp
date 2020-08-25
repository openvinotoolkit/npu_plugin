// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef __unix__
#include <elf.h>
#else
#include <cstdint>
__pragma(pack(push, 1))
struct Elf32_Ehdr {
    uint8_t  offs1[28];
    uint32_t e_phoff;        // Program header offset
    uint32_t e_shoff;        // Section header offset
    uint8_t  offs2[12];
    uint16_t e_shnum;        // Number of sections
    uint16_t e_shstrndx;     // String table index
};

struct Elf32_Shdr {
    uint32_t sh_name;        // Section name index
    uint32_t sh_type;
    uint32_t sh_flags;
    uint32_t sh_addr;        // Section virtual address
    uint32_t sh_offset;      // Section file offset
    uint32_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint32_t sh_addralign;   // Section alignment
    uint32_t sh_entsize;
};
__pragma(pack(pop))
#endif

#include <xml_parse_utils.h>

#include <custom_layer/ShaveElfMetadataParser.hpp>
#include <custom_layer/custom_kernel.hpp>
#include <caseless.hpp>
#include <vpu/utils/error.hpp>
#include <vpu/utils/extra.hpp>

namespace vpu {

static const Elf32_Shdr *get_elf_section_with_name(const uint8_t *elf_data, const char* section_name) {
    IE_ASSERT(elf_data);
    IE_ASSERT(section_name);

    const Elf32_Ehdr *ehdr = reinterpret_cast<const Elf32_Ehdr *>(elf_data);
    IE_ASSERT(0 != ehdr->e_shoff);
    IE_ASSERT(0 != ehdr->e_phoff);

    // Pointer to the first section header
    const Elf32_Shdr *shdr = reinterpret_cast<const Elf32_Shdr *>(elf_data + ehdr->e_shoff);

    // Pointer to section header string table header
    const Elf32_Shdr *strShdr = &shdr[ehdr->e_shstrndx];

    // We couldn't find sections for the symbol string names and for the symbols
    // entries
    if (!strShdr) {
        return nullptr;
    }

    // The string at index 0, which corresponds to the first byte, is a null
    // character
    const char *firstStr = reinterpret_cast<const char *>(elf_data + strShdr->sh_offset);

    // Find the section with the custom SHAVEComputeAorta data
    for (decltype(ehdr->e_shnum) i = 0; i < ehdr->e_shnum; i++) {
        const char *currentSectionName = firstStr + shdr[i].sh_name;

        if (0 == strcmp(currentSectionName, section_name)) {
            return shdr + i;
        }
    }

    // If we reached this point, it means that there wasn't a section with
    // the name we were looking for
    return nullptr;
}

SmallVector<CustomKernel::Argument> deduceKernelArguments(const md_parser_t& parser, int kernelId) {
    const auto kernelDesc = parser.get_kernel(kernelId);
    IE_ASSERT(kernelDesc != nullptr);

    // compiler workaround
    const auto argCount = kernelDesc->arg_count - 1;

    const auto nextName = [](const char* name) {
        while (*++name) {}
        return name + 1;
    };

    auto namePtr = parser.get_string_table();
    // first name in table is kernel name
    namePtr = nextName(namePtr);

    auto arguments = SmallVector<CustomKernel::Argument>{};
    arguments.reserve(argCount);
    for (size_t i = 0; i < argCount; i++) {
        const auto arg = parser.get_argument(kernelDesc, i);
        VPU_THROW_UNLESS(arg, "Cant find argument number %l", i);
        // FIXME use get_name after compiler fix
//        const auto argName = parser.get_name(arg);
        arguments.emplace_back(std::string{namePtr}, static_cast<int>(arg->size_elm));
        namePtr = nextName(namePtr);
    }

    return arguments;
}

CustomKernel::CustomKernel(const pugi::xml_node& kernel, std::string configDir): _configDir {std::move(configDir)} {
    _maxShaves = XMLParseUtils::GetIntAttr(kernel, "max-shaves", 0);

    for (auto source = kernel.child("Source"); !source.empty(); source = source.next_sibling("Source")) {
        auto fileName = _configDir + "/" + XMLParseUtils::GetStrAttr(source, "filename", "");

        std::ifstream inputFile(fileName, std::ios::in | std::ios::binary);
        if (!inputFile.is_open()) {
            THROW_IE_EXCEPTION << "Couldn't open kernel file " << fileName;
        }

        std::ostringstream contentStream;
        contentStream << inputFile.rdbuf();
        const auto kernelData = contentStream.str();

        _kernelBinary.insert(end(_kernelBinary), begin(kernelData), end(kernelData));
    }

    const auto kernelEntryName = XMLParseUtils::GetStrAttr(kernel, "entry");

    const auto elf = _kernelBinary.data();
    const Elf32_Shdr *neoMetadataShdr = get_elf_section_with_name(elf, ".neo_metadata");
    VPU_THROW_UNLESS(neoMetadataShdr, "Error while parsing custom layer elf: Couldn't find .neo_metadata section");

    const uint8_t *neoMetadata = elf + neoMetadataShdr->sh_offset;
    const size_t neoMetadataSize = neoMetadataShdr->sh_size;

    const Elf32_Shdr *neoMetadataStrShdr = get_elf_section_with_name(elf, ".neo_metadata.str");
    VPU_THROW_UNLESS(neoMetadataStrShdr,"Error while parsing custom layer elf: Couldn't find .neo_metadata.str section");

    const char *neoMetadataStr = reinterpret_cast<const char *>(elf + neoMetadataStrShdr->sh_offset);
    const size_t neoMetadataStrSize = neoMetadataStrShdr->sh_size;

    const auto parser = md_parser_t{neoMetadata, neoMetadataSize, neoMetadataStr, neoMetadataStrSize};
    _kernelId = parser.get_kernel_id(kernelEntryName);
    _kernelArguments = deduceKernelArguments(parser, _kernelId);

    processParametersNode(kernel);
    processWorkSizesNode(kernel);

    const auto isInputData = [&](const std::pair<std::string, CustomKernel::BindingParameter>& binding) {
        const auto& param = binding.second;
        return param.type == CustomParamType::Input || param.type == CustomParamType::InputBuffer ||
               param.type == CustomParamType::Data;
    };

    _inputDataCount = std::count_if(begin(_bindings), end(_bindings), isInputData);
}

std::pair<CustomDimSource, int> parseDimSource(const std::string& dims) {
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


CustomDataFormat formatFromString(const std::string& str) {
    static const ie::details::caseless_map<std::string, CustomDataFormat> FormatNameToType = {
        { "BFYX" , CustomDataFormat::BFYX },
        { "BYXF" , CustomDataFormat::BYXF },
        { "FYX" , CustomDataFormat::FYX },
        { "YXF" , CustomDataFormat::YXF },
        { "BF" , CustomDataFormat::BF },
        { "ANY"  , CustomDataFormat::Any }
    };

    auto it = FormatNameToType.find(str);
    if (it != FormatNameToType.end()) {
        return it->second;
    }

    THROW_IE_EXCEPTION << "Tensor node has an invalid format '" << str << "'";
}

SmallVector<std::string> parseSizeRule(const std::string& size) {
    auto result = SmallVector<std::string>();
    result.reserve(std::count(begin(size), end(size), ',') + 1);
    std::stringstream sizeRules{size};
    std::string bufferSize;

    while (std::getline(sizeRules, bufferSize, ',')) {
        result.push_back(bufferSize);
    }

    return result;
}

void CustomKernel::processParametersNode(const pugi::xml_node& node) {
    const auto cmp = ie::details::CaselessEq<std::string> {};
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

    for (auto& binding : bindings) {
        _bindings[binding.argName] = std::move(binding);
    }
}

void CustomKernel::processWorkSizesNode(const pugi::xml_node& node) {
    const auto workSizes = node.child("WorkSizes");

    const auto dims = XMLParseUtils::GetStrAttr(workSizes, "dim");
    std::tie(_wgDimSource, _wgDimIdx) = parseDimSource(dims);

    const auto gwgs = XMLParseUtils::GetStrAttr(workSizes, "global");
    _globalGridSizeRules = parseSizeRule(gwgs);

    const auto lwgs = XMLParseUtils::GetStrAttr(workSizes, "local");
    _localGridSizeRules = parseSizeRule(lwgs);
}

} // namespace vpu

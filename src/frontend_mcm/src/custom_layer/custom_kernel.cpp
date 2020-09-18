// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <xml_parse_utils.h>
#include <custom_layer/ShaveElfMetadataParser.hpp>
#include <custom_layer/custom_kernel.hpp>
#include <caseless.hpp>
#include <vpu/utils/error.hpp>
#include <vpu/utils/extra.hpp>

namespace vpu {

VPU_PACKED(Elf32Shdr {
    uint32_t shName;
    uint32_t pad0[3];
    uint32_t shOffset;
    uint32_t shSize;
    uint32_t pad1[4];
};)

VPU_PACKED(Elf32Ehdr {
    uint32_t pad0[7];
    uint32_t ePhoff;
    uint32_t eShoff;
    uint32_t pad1[3];
    uint16_t eShnum;
    uint16_t eShstrndx;
};)

VPU_PACKED(Elf32Section {
    uint32_t shName;
    uint32_t shType;
    uint32_t shFlags;
    uint32_t shAddr;
    uint32_t shOffset;
    uint32_t shSize;
    uint32_t shLink;
    uint32_t shInfo;
    uint32_t shAddralign;
    uint32_t shEntsize;
};)

VPU_PACKED(Elf32Phdr {
    uint32_t pType;       // Identifies program segment type
    uint32_t pOffset;     // Segment file offset
    uint32_t pVaddr;      // Segment virtual address
    uint32_t pPaddr;      // Segment physical address
    uint32_t pFilesz;     // Segment size in file
    uint32_t pMemsz;      // Segment size in memory
    uint32_t pFlags;      // Flags position from ELF standard spec
    uint32_t pAlign;      // Segment alignment, file & memory
};)

VPU_PACKED(Elf32Sym {
    uint32_t stName;
    uint32_t stValue;
    uint32_t stSize;
    uint8_t  stInfo;
    uint8_t  stOther;
    uint16_t stShndx;
};)

VPU_PACKED(KernelHdr {
    uint32_t address;       // Kernel address
    uint32_t flags;         // Should be 0 for now
    uint32_t sectionSize;   // Section size, offset to the next kernel
    uint32_t argOffset;     // offset to arguments
    uint32_t stackSize;     // Size of the stack required for kernel
    uint32_t stackSizeWI;     // Size of the stack required for kernel per WI
};)

VPU_PACKED(KernelArgHdr {
    uint32_t stringOffset;
    uint32_t addressSpace;
    uint32_t typeOffset;
    uint32_t size;
    uint32_t laneSize;
};)

static const Elf32Shdr *get_elf_section_with_name(const uint8_t *elf_data, const char* section_name) {
    IE_ASSERT(elf_data);
    IE_ASSERT(section_name);

    const auto *ehdr = reinterpret_cast<const Elf32Ehdr *>(elf_data);
    IE_ASSERT(0 != ehdr->eShoff);
    IE_ASSERT(0 != ehdr->ePhoff);

    // Pointer to the first section header
    const Elf32Shdr *shdr = reinterpret_cast<const Elf32Shdr *>(elf_data + ehdr->eShoff);

    // Pointer to section header string table header
    const Elf32Shdr *strShdr = &shdr[ehdr->eShstrndx];

    // We couldn't find sections for the symbol string names and for the symbols
    // entries
    if (!strShdr) {
        return nullptr;
    }

    // The string at index 0, which corresponds to the first byte, is a null
    // character
    const char *firstStr = reinterpret_cast<const char *>(elf_data + strShdr->shOffset);

    // Find the section with the custom SHAVEComputeAorta data
    for (uint16_t i = 0; i < ehdr->eShnum; i++) {
        const char *currentSectionName = firstStr + shdr[i].shName;

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
    // Number of elements we get from parser is always greater by one
    const auto argCount = kernelDesc->arg_count - 1;

    auto arguments = SmallVector<CustomKernel::Argument>{};
    arguments.reserve(argCount);
    for (size_t i = 0; i < argCount; i++) {
        const auto arg = parser.get_argument(kernelDesc, i);
        VPU_THROW_UNLESS(arg, "Error while parsing custom layer elf file.");
        const auto argName = parser.get_name(arg);
        arguments.emplace_back(argName, static_cast<int>(arg->size_elm));
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
    const Elf32Shdr *neoMetadataShdr = get_elf_section_with_name(elf, ".neo_metadata");
    VPU_THROW_UNLESS(neoMetadataShdr, "Error while parsing custom layer elf: Couldn't find .neo_metadata section");

    const uint8_t *neoMetadata = elf + neoMetadataShdr->shOffset;
    const size_t neoMetadataSize = neoMetadataShdr->shSize;

    const Elf32Shdr *neoMetadataStrShdr = get_elf_section_with_name(elf, ".neo_metadata.str");
    VPU_THROW_UNLESS(neoMetadataStrShdr,"Error while parsing custom layer elf: Couldn't find .neo_metadata.str section");

    const char *neoMetadataStr = reinterpret_cast<const char *>(elf + neoMetadataStrShdr->shOffset);
    const size_t neoMetadataStrSize = neoMetadataStrShdr->shSize;

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

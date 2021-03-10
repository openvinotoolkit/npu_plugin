// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_layer/custom_kernel.hpp"

#include "custom_layer/ShaveElfMetadataParser.hpp"

#include "vpux/utils/core/error.hpp"

#include <xml_parse_utils.h>
#include <caseless.hpp>
#include <vpu/utils/extra.hpp>

namespace vpu {

struct Argument final {
    std::string name;
    uint32_t typeSize;
};

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

static const Elf32Shdr* get_elf_section_with_name(const uint8_t* elf_data, const char* section_name) {
    IE_ASSERT(elf_data);
    IE_ASSERT(section_name);

    const auto* ehdr = reinterpret_cast<const Elf32Ehdr*>(elf_data);
    IE_ASSERT(0 != ehdr->eShoff);
    IE_ASSERT(0 != ehdr->ePhoff);

    // Pointer to the first section header
    const Elf32Shdr* shdr = reinterpret_cast<const Elf32Shdr*>(elf_data + ehdr->eShoff);

    // Pointer to section header string table header
    const Elf32Shdr* strShdr = &shdr[ehdr->eShstrndx];

    // We couldn't find sections for the symbol string names and for the symbols
    // entries
    if (!strShdr) {
        return nullptr;
    }

    // The string at index 0, which corresponds to the first byte, is a null
    // character
    const char* firstStr = reinterpret_cast<const char*>(elf_data + strShdr->shOffset);

    // Find the section with the custom SHAVEComputeAorta data
    for (uint16_t i = 0; i < ehdr->eShnum; i++) {
        const char* currentSectionName = firstStr + shdr[i].shName;

        if (0 == strcmp(currentSectionName, section_name)) {
            return shdr + i;
        }
    }

    // If we reached this point, it means that there wasn't a section with
    // the name we were looking for
    return nullptr;
}

SmallVector<Argument> deduceKernelArguments(const md_parser_t& parser, int kernelId) {
    const auto kernelDesc = parser.get_kernel(kernelId);
    IE_ASSERT(kernelDesc != nullptr);
    // Number of elements we get from parser is always greater by one
    const auto argCount = kernelDesc->arg_count - 1;

    auto arguments = SmallVector<Argument>{};
    arguments.reserve(argCount);
    for (size_t i = 0; i < argCount; i++) {
        const auto arg = parser.get_argument(kernelDesc, i);
        VPUX_THROW_UNLESS(arg, "Error while parsing custom layer elf file.");
        const auto argName = parser.get_name(arg);

        // skip hoisted buffers
        if (arg->flags & md_arg_flags_generated_prepost) {
            continue;
        }

        arguments.emplace_back(Argument{argName, arg->size_elm});
    }

    return arguments;
}

md_parser_t createParser(const std::vector<uint8_t>& kernelBinary) {
    const auto elf = kernelBinary.data();
    const Elf32Shdr* neoMetadataShdr = get_elf_section_with_name(elf, ".neo_metadata");
    VPUX_THROW_UNLESS(neoMetadataShdr, "Error while parsing custom layer elf: Couldn't find .neo_metadata section");

    const uint8_t* neoMetadata = elf + neoMetadataShdr->shOffset;
    const size_t neoMetadataSize = neoMetadataShdr->shSize;

    const Elf32Shdr* neoMetadataStrShdr = get_elf_section_with_name(elf, ".neo_metadata.str");
    VPUX_THROW_UNLESS(neoMetadataStrShdr,
                      "Error while parsing custom layer elf: Couldn't find .neo_metadata.str section");

    const char* neoMetadataStr = reinterpret_cast<const char*>(elf + neoMetadataStrShdr->shOffset);
    const size_t neoMetadataStrSize = neoMetadataStrShdr->shSize;

    const auto parser = md_parser_t{neoMetadata, neoMetadataSize, neoMetadataStr, neoMetadataStrSize};
    return parser;
}

CustomKernelOcl::CustomKernelOcl(const pugi::xml_node& node, const std::string& configDir) {
    _maxShaves = XMLParseUtils::GetIntAttr(node, "max-shaves", 0);
    _kernelBinary = loadKernelBinary(node, configDir);

    processWorkSizesNode(node);

    md_parser_t parser = createParser(_kernelBinary);
    const auto kernelEntryName = XMLParseUtils::GetStrAttr(node, "entry");
    _kernelId = parser.get_kernel_id(kernelEntryName);
    VPUX_THROW_UNLESS(_kernelId != -1, "Failed to find kernel with name `{0}`", kernelEntryName);

    VPUX_THROW_UNLESS(parser.get_kernel_count() == 1,
                      "Failed to load kernel binary\n"
                      "\tReason: binary should contain only one kernel, but contains {0}",
                      parser.get_kernel_count());

    auto arguments = deduceKernelArguments(parser, _kernelId);
    auto bindings = processParametersNode(node);

    for (const auto& argument : arguments) {
        const auto withBindingName = [&](const BindingParameter& bind) {
            return bind.argName == argument.name;
        };

        auto binding = std::find_if(begin(bindings), end(bindings), withBindingName);
        IE_ASSERT(binding != bindings.end());

        if (binding->type == CustomParamType::Output && argument.typeSize != 1 && argument.typeSize != 2) {
            VPUX_THROW("Custom layer output parameter '{0}' has unsupported output data type "
                       "with underlying type size = {1}",
                       argument.name, argument.typeSize);
        }

        _kernelBindings.push_back(*binding);
    }

    const auto isInputData = [&](const CustomKernel::BindingParameter& param) {
        return param.type == CustomParamType::Input || param.type == CustomParamType::InputBuffer ||
               param.type == CustomParamType::Data;
    };

    _inputDataCount = std::count_if(begin(_kernelBindings), end(_kernelBindings), isInputData);
}

void CustomKernelOcl::accept(CustomKernelVisitor& validator) const {
    validator.visitCL(*this);
}

void CustomKernelOcl::processWorkSizesNode(const pugi::xml_node& node) {
    const auto workSizes = node.child("WorkSizes");

    const auto dims = XMLParseUtils::GetStrAttr(workSizes, "dim");
    std::tie(_wgDimSource, _wgDimIdx) = parseDimSource(dims);

    const auto gwgs = XMLParseUtils::GetStrAttr(workSizes, "global");
    _globalGridSizeRules = parseSizeRule(gwgs);

    const auto lwgs = XMLParseUtils::GetStrAttr(workSizes, "local");
    _localGridSizeRules = parseSizeRule(lwgs);
}

}  // namespace vpu

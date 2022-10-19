//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/ELF/utils.hpp"
#include <vpux_elf/reader.hpp>

std::pair<uint8_t*, size_t> vpux::ELF::getDataAndSizeOfElfSection(const std::vector<char>& elfBlob,
                                                                  const std::vector<std::string> possibleSecNames) {
    auto elf_reader = elf::Reader<elf::ELF_Bitness::Elf32>((const uint8_t*)elfBlob.data(), elfBlob.size());

    uint8_t* secData;
    uint32_t secSize = 0;

    bool secFound = false;

    for (size_t i = 0; i < elf_reader.getSectionsNum(); ++i) {
        auto section = elf_reader.getSection(i);
        const auto secName = section.getName();
        const auto sectionHeader = section.getHeader();

        for (auto possibleSecName : possibleSecNames) {
            if (strcmp(secName, possibleSecName.c_str()) == 0) {
                secSize = sectionHeader->sh_size;
                secData = (uint8_t*)section.getData<uint8_t>();
                secFound = true;
                break;
            }
        }
    }
    VPUX_THROW_UNLESS(secFound, "Section {0} not found in ELF", possibleSecNames);

    return {secData, secSize};
}

//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <vpux_elf/utils/utils.hpp>

#include <vpux_elf/types/elf_header.hpp>

#include <vpux_elf/utils/error.hpp>

using namespace elf;

void utils::checkELFMagic(const unsigned char* elfIdent) {
    if (elfIdent[elf::EI_MAG0] != elf::ELFMAG0 ||
        elfIdent[elf::EI_MAG1] != elf::ELFMAG1 ||
        elfIdent[elf::EI_MAG2] != elf::ELFMAG2 ||
        elfIdent[elf::EI_MAG3] != elf::ELFMAG3) {
        VPUX_ELF_THROW("Incorrect ELF magic");
    }
}

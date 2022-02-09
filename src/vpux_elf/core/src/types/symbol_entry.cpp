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

#include <vpux_elf/types/symbol_entry.hpp>

using namespace elf;

//! Extract symbol binding attributes from info
Elf_Xword elf::elf64STBind(Elf_Xword info) {
    return info >> 4;
}

//! Extract symbol type from info
Elf_Xword elf::elf64STType(Elf_Xword info) {
    return info & 0xf;
}

//! Pack symbol binding attributes and symbol type into info
Elf_Xword elf::elf64STInfo(Elf_Word bind, Elf_Word type) {
    return (bind << 4) + (type & 0xf);
}

//! Performs a transformation over visibility to zero out all bits that have no defined meaning
uint8_t elf::elf64STVisibility(uint8_t visibility) {
    return visibility & 0x3;
}

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

#include <vpux_elf/types/relocation_entry.hpp>

using namespace elf;

//! Extract symbol index from info
Elf_Word elf::elf64RSym(Elf_Xword info) {
    return info >> 32;
}

//! Extract relocation type from info
Elf_Word elf::elf64RType(Elf_Xword info) {
    return static_cast<Elf_Word>(info);
}

//! Pack relocation type and symbol index into info
Elf_Xword elf::elf64RInfo(Elf_Word sym, Elf_Word type) {
    return (static_cast<Elf_Xword>(sym) << 32) + (static_cast<Elf_Xword>(type));
}

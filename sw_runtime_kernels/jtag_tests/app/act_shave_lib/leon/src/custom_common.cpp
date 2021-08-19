/*
* {% copyright %}
*/
#include "custom_common.h"

#include <nn_cache.h>
#include <nn_cache.h>
#include <mv_types.h>

#include <cstring>

namespace nn {
namespace shave_lib {

const Elf32_Shdr *get_elf_section_with_name(const uint8_t *elf_data, const char *section_name) {
    RETURN_NULL_UNLESS(elf_data);
    RETURN_NULL_UNLESS(section_name);

    const Elf32_Ehdr *ehdr = reinterpret_cast<const Elf32_Ehdr *>(elf_data);
    RETURN_NULL_UNLESS(0 != ehdr->e_shoff);
    RETURN_NULL_UNLESS(0 != ehdr->e_phoff);

    // Pointer to the first section header
    const Elf32_Shdr *shdr = reinterpret_cast<const Elf32_Shdr *>(elf_data + ehdr->e_shoff);

    // Pointer to section header string table header
    const Elf32_Shdr *strShdr = &shdr[ehdr->e_shstrndx];

    // We couldn't find sections for the symbol string names and for the symbols
    // entries
    RETURN_NULL_UNLESS(strShdr);

    // The string at index 0, which corresponds to the first byte, is a null
    // character
    const char *firstStr = reinterpret_cast<const char *>(elf_data + strShdr->sh_offset);

    // Find the section with the custom SHAVEComputeAorta data
    for (Elf32_Half i = 0; i < ehdr->e_shnum; i++) {
        const char *currentSectionName = firstStr + shdr[i].sh_name;

        logI("Current section name: %s", currentSectionName);
        if (0 == strcmp(currentSectionName, section_name)) {
            return shdr + i;
        }
    }

    // If we reached this point, it means that there wasn't a section with
    // the name we were looking for
    return nullptr;
}

bool loadElf(const uint8_t *elfAddr, void *buffer) {
    const Elf32_Ehdr *elfHeader = reinterpret_cast<const Elf32_Ehdr *>(elfAddr);

    // Make sure this is a valid ELF header
    auto elfMagic = elfHeader->e_ident[0] == 0x7F &&
                    elfHeader->e_ident[1] == 'E' &&
                    elfHeader->e_ident[2] == 'L' &&
                    elfHeader->e_ident[3] == 'F';

    RETURN_FALSE_UNLESS(elfMagic, "Failed to load unsupported ELF file");

    // Reading section headers table offset
    u32 phAddr = (u32)elfAddr + elfHeader->e_shoff;

    // Parse section headers:
    for (u32 SecHeaders = 0; SecHeaders < elfHeader->e_shnum; SecHeaders++) {
        u32 SecOffset = phAddr + sizeof(Elf32_Shdr) * SecHeaders;

        const Elf32_Shdr *ElfSecHeader = reinterpret_cast<const Elf32_Shdr *>(SecOffset);

        // Only load PROGBITS sections
        // Our generated ELF files only have two sections - 1 code and 1 data
        if (ElfSecHeader->sh_type == SHT_PROGBITS) {
            // Executable (code) section
            u32 *srcAddr = (u32 *)((u32)elfAddr + ElfSecHeader->sh_offset);

            u32 inAddr = ElfSecHeader->sh_addr;
            u32 window = (inAddr >> 24) - 0x1C;

            RETURN_FALSE_UNLESS(window == 2 || window == 3,
                                "Unsupported binary detected: contains A or B windowed code!");

            // local and private memory for OpenCL is not zero initialized, so we can skip relocation of this section of
            // OpenCL generated bits
            if (window == 3)
                continue;

            // it actually copies only code window since there should not be absolute addresses in the OpenCL generated
            // file
            u32 *dstAddr = (u32 *)((inAddr & 0x00FFFFFF) + (u32)buffer);

            logI("copy ElfSecHeader->sh_type=%lu:  \n"
                 "srcAddr=%p dstAddr=%p ElfSecHeader->sh_addr=%x ElfSecHeader->sh_size=%u window=%u",
                 ElfSecHeader->sh_type, srcAddr, dstAddr,
                 ElfSecHeader->sh_addr, ElfSecHeader->sh_size, window);

            memcpy_s(dstAddr, ElfSecHeader->sh_size, srcAddr, ElfSecHeader->sh_size);
            nn::cache::flush(dstAddr, ElfSecHeader->sh_size);
        }
    }
    return true;
}

} // namespace shave_lib
} // namespace nn

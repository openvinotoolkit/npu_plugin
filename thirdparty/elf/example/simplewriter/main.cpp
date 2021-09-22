// {% copyright %}

#include <elf/elf.hpp>

int main() {
    elf::ELF elf{};
    auto section = elf.addSection();
    section->setType(elf::SHT_PROGBITS);
    elf.writeTo("blob.elf");

    return 0;
}

// {% copyright %}

#include <elf/elf.hpp>

int main() {
    elf::ELF elf{};
    elf.readFrom("elf.blob");

    return 0;
}

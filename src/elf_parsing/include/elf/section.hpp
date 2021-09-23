// {% copyright %}

#pragma once

#include <vector>

namespace elf {

template<typename T, template<typename> class Alloc>
class Section {
public:
    Section() = default;

    Elf_Half getType() { return _sectionHeader.sh_type; }
    void setType(Elf_Half type) { _sectionHeader.sh_type = type; }

private:
    T _sectionHeader;
    std::vector<char, Alloc<char>> _data;
};

} // namespace elf

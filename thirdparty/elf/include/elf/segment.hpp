// {% copyright %}

#pragma once

#include <vector>

namespace elf {

template<typename T, template<typename> class Alloc>
class Segment {
public:
    Segment() = default;

    Elf_Half getType() { return _programHeader.p_type; }
    void setType(Elf_Half type) { _programHeader.p_type = type; }

private:
    T _programHeader;
    std::vector<char, Alloc<char>> _data;
};

} // namespace elf

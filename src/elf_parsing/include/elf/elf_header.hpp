// {% copyright %}

#pragma once

#include <elf/types/elf_header.hpp>
#include <elf/types/data_types.hpp>

#include <memory>
#include <cstring>

namespace elf {

template<typename T>
class ElfHeader {
public:
    void readFrom(const char* data) {
        std::memcpy(&_header, data, sizeof(T));
    }

    Elf_Half getType() { return _header->e_type; }
    void setType(Elf_Half type) { _header->e_type = type; }

private:
    T _header;
};

} // namespace elf

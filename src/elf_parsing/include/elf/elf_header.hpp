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

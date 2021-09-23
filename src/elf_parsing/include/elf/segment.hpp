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

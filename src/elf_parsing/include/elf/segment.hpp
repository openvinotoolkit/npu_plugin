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

    Elf_Half getType() const { return _programHeader.sh_type; }
    void setType(Elf_Half type) { _programHeader.sh_type = type; }

private:
    T _programHeader;
    std::vector<char, Alloc<char>> _data;
};

template<typename T>
class SegmentRef {
public:
    SegmentRef() = delete;
    SegmentRef(const char* programHeader, const char* data) {
        _programHeader = reinterpret_cast<const T*>(programHeader);
        _data = data;
    }

    Elf_Half getType() const { return _programHeader->sh_type; }

private:
    const T* _programHeader;
    const char* _data;
};

} // namespace elf

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

#include <elf/elf_header.hpp>

#include <elf/section.hpp>
#include <elf/segment.hpp>

#include <elf/utils/traits.hpp>
#include <elf/utils/utils.hpp>

#include <string>
#include <vector>
#include <memory>

namespace elf {

/**
 * @brief This class represents an ELF file
 */

template<Elf_Half Class, template<typename> class Alloc = std::allocator>
class GenericELF {
    using SectionT = Section<typename HeaderTypes<Class>::Elf_SHdr, Alloc>;
    using SegmentT = Segment<typename HeaderTypes<Class>::Elf_PHdr, Alloc>;

public:
    SectionT* addSection() {
        // TODO: it's not correct to return pointer to vector element
        _sections.emplace_back();
        return &(_sections.back());
    }

    SegmentT* addSegment() {
        _segments.emplace_back();
        return &(_segments.back());
    }

    void readFrom(const std::string& fileName) {
        std::ifstream stream(fileName, std::ios::binary);
        readFrom(stream);
    }

    void readFrom(std::istream& stream) {
        const auto dataSize = utils::getDataSize(stream);
        std::vector<char, Alloc<char>> elf(dataSize);
        stream.read(elf.data(), dataSize);
        readFrom(std::move(elf));
    }

    void readFrom(std::vector<char>) {
        // TODO: not implemented
    }

    void writeTo(const std::string&) {
        // TODO: not implemented
    }

    void writeTo(std::ostream&) {
        // TODO: not implemented
    }

private:
    ElfHeader<typename HeaderTypes<Class>::Elf_EHdr> _elfHeader;
    std::vector<SegmentT, Alloc<SegmentT>> _segments;
    std::vector<SectionT, Alloc<SectionT>> _sections;
};

//! 64-bit ELF with standard allocator
using ELF = GenericELF<ELF64, std::allocator>;

} // namespace elf

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

#include <elf/types/data_types.hpp>

#include <elf/section.hpp>
#include <elf/segment.hpp>

#include <elf/utils/traits.hpp>

#include <string>

namespace elf {

template<Elf_Half Class, template<typename> class Alloc = std::allocator>
class GenericELFReader {
    using SectionT = SectionRef<typename HeaderTypes<Class>::Elf_SHdr>;
    using SegmentT = SegmentRef<typename HeaderTypes<Class>::Elf_PHdr>;

public:
    GenericELFReader(const std::string&) {
        // TODO: implement
    }
    GenericELFReader(std::istream&) {
        // TODO: implement
    }
    GenericELFReader(std::vector<char>) {
        // TODO: implement
    }

    Elf_Half getType() const { return _elfHeader->e_type; }

private:
    std::vector<char> _blob;

    const typename HeaderTypes<Class>::Elf_EHdr* _elfHeader;
    std::vector<SegmentT, Alloc<SegmentT>> _segments;
    std::vector<SectionT, Alloc<SectionT>> _sections;
};

//! 64-bit ELF reader with standard allocator
using ELFReader = GenericELFReader<ELF64, std::allocator>;

} // namespace elf
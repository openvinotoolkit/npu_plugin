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
#include <elf/types/section_header.hpp>
#include <elf/types/program_header.hpp>
#include <elf/types/elf_header.hpp>

#include <string>
#include <vector>

namespace elf {

class Reader {
public:
    explicit Reader(const char* blob);

    Elf_Half getType() const;

public:
    class Section {
    public:
        Section() = delete;
        Section(const char* sectionHeader, const char* data);

        Elf_Half getType() const;

    private:
        const Elf64_Shdr* m_sectionHeader;
        const char* m_data;
    };

    class Segment {
    public:
        Segment() = delete;
        Segment(const char* programHeader, const char* data);

        Elf_Half getType() const;

    private:
        const Elf64_Phdr* m_programHeader;
        const char* m_data;
    };

private:
    Elf64_Ehdr* m_elfHeader;
    std::vector<Segment> m_segments;
    std::vector<Section> m_sections;
};

} // namespace elf
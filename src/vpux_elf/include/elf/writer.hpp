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

#include <elf/writer/section.hpp>
#include <elf/writer/segment.hpp>

#include <elf/writer/relocation_section.hpp>
#include <elf/writer/symbol_section.hpp>
#include <elf/writer/binary_data_section.hpp>
#include <elf/writer/string_section.hpp>
#include <elf/writer/empty_section.hpp>

#include <elf/types/data_types.hpp>
#include <elf/types/section_header.hpp>
#include <elf/types/program_header.hpp>
#include <elf/types/elf_header.hpp>

#include <string>
#include <vector>

namespace elf {

class Writer {
public:
    Writer();

    std::vector<char> generateELF();

    writer::Segment* addSegment();

    writer::RelocationSection* addRelocationSection();
    writer::SymbolSection* addSymbolSection();
    writer::EmptySection* addEmptySection();

    template <typename T>
    writer::BinaryDataSection<T>* addBinaryDataSection() {
        m_sections.push_back(std::unique_ptr<writer::BinaryDataSection<T>>(new writer::BinaryDataSection<T>));
        m_sections.back()->setIndex(m_sections.size() - 1);
        return dynamic_cast<writer::BinaryDataSection<T>*>(m_sections.back().get());
    }

private:
    writer::Section* addSection();
    writer::StringSection* addStringSection();

    elf::ELFHeader generateELFHeader() const;

private:
    writer::StringSection* m_sectionHeaderNames;
    writer::StringSection* m_symbolNames;
    std::vector<writer::Section::Ptr> m_sections;
    std::vector<writer::Segment::Ptr> m_segments;
};

} // namespace elf
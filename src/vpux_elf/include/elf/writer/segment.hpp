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
#include <elf/types/program_header.hpp>

#include <elf/writer/section.hpp>

#include <vector>
#include <memory>

namespace elf {

class Writer;

namespace writer {

class Segment {
public:
    using Ptr = std::unique_ptr<Segment>;

public:
    template <typename T>
    void addData(const T* data, size_t size) {
        m_data.insert(m_data.end(), reinterpret_cast<const char*>(data), reinterpret_cast<const char*>(data) + size);
    }
    void addSection(Section* section);

    void setType(Elf_Word type);
    void setAlign(Elf_Xword align);

private:
    Segment();

    ProgramHeader m_header{};

    std::vector<char> m_data;
    std::vector<Section*> m_sections;

    friend Writer;
};

} // namespace writer
} // namespace elf

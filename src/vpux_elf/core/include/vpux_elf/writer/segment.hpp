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

#include <vpux_elf/types/data_types.hpp>
#include <vpux_elf/types/program_header.hpp>

#include <vpux_elf/writer/section.hpp>

#include <vector>
#include <memory>

namespace elf {

class Writer;

namespace writer {

class Segment {
public:
    template <typename T>
    void appendData(const T* data, size_t sizeInElements) {
        m_data.insert(m_data.end(), reinterpret_cast<const uint8_t*>(data), reinterpret_cast<const uint8_t*>(data) + sizeInElements  * sizeof(T));
    }
    void addSection(Section* section);

    void setType(Elf_Word type);
    void setAlign(Elf_Xword align);

private:
    using Ptr = std::unique_ptr<Segment>;

private:
    Segment();

    ProgramHeader m_header{};

    std::vector<char> m_data;
    std::vector<Section*> m_sections;

    friend Writer;
};

} // namespace writer
} // namespace elf

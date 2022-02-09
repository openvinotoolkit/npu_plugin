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
#include <vpux_elf/types/section_header.hpp>

#include <vector>
#include <string>
#include <memory>

namespace elf {

class Writer;

namespace writer {

class Section {
public:
    std::string getName() const;
    void setName(const std::string& name);

    Elf_Xword getAddrAlign() const;
    void setAddrAlign(Elf_Xword addrAlign);

    Elf64_Addr getAddr() const;
    void setAddr(Elf64_Addr addr);

    Elf_Xword getFlags() const;
    void setFlags(Elf_Xword flags);
    void maskFlags(Elf_Xword flags);

    size_t getFileAlignRequirement() const;

    size_t getIndex() const;
    size_t getDataSize() const;

    virtual ~Section() = default;

protected:
    explicit Section(const std::string& name = {});

    virtual void finalize();

    void setIndex(size_t index);
    void setNameOffset(size_t offset);

protected:
    std::string m_name;
    size_t m_index = 0;
    size_t m_fileAlignRequirement = 1;

    SectionHeader m_header{};
    std::vector<uint8_t> m_data;

    friend Writer;
};

} // namespace writer
} // namespace elf

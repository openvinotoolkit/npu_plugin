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

#include <vpux_elf/writer/string_section.hpp>

using namespace elf;
using namespace elf::writer;

StringSection::StringSection() {
    m_header.sh_type = SHT_STRTAB;
    m_data.push_back('\0');
}

size_t StringSection::addString(const std::string& name) {
    if (name.empty()) {
        return 0;
    }

    const auto pos = m_data.size();
    const auto str = name.c_str();
    m_data.insert(m_data.end(), str, str + name.size() + 1);

    return pos;
}

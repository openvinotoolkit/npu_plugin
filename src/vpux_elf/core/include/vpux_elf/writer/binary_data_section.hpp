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

#include <vpux_elf/writer/section.hpp>

namespace elf {

class Writer;

namespace writer {

template <typename T>
class BinaryDataSection final : public Section {
public:
    size_t appendData(const T& obj) {
        return appendData(&obj, 1);
    }

    size_t appendData(const T* obj, size_t sizeInElements) {
        const auto offset = m_data.size();
        m_data.insert(m_data.end(), reinterpret_cast<const uint8_t*>(obj),
                      reinterpret_cast<const uint8_t*>(obj) + sizeInElements * sizeof(T));
        return offset;
    }

    size_t getNumEntries() const {
        return static_cast<size_t>(m_data.size() / sizeof(T));
    }

private:
    explicit BinaryDataSection(const std::string& name) : Section(name) {
        static_assert(std::is_standard_layout<T>::value, "Only POD types are supported");
        m_header.sh_type = SHT_PROGBITS;
        m_header.sh_entsize = sizeof(T);
    }

    friend Writer;
};

} // namespace writer
} // namespace elf

// {% copyright %}

#pragma once

#include <fstream>

namespace elf {
namespace utils {

inline size_t getDataSize(std::istream& strm) {
    const size_t streamStart = strm.tellg();
    strm.seekg(0, std::ios_base::end);
    const size_t streamEnd = strm.tellg();
    const size_t bytesAvailable = streamEnd - streamStart;
    strm.seekg(streamStart, std::ios_base::beg);

    return bytesAvailable;
}

} // namespace utils
} // namespace elf

//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/file_reader.hpp"

#include <array>
#include <fstream>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace vpux {

size_t getFileSize(std::istream& strm) {
    const size_t streamStart = strm.tellg();
    strm.seekg(0, std::ios_base::end);
    const size_t streamEnd = strm.tellg();
    const size_t bytesAvailable = streamEnd - streamStart;
    strm.seekg(streamStart, std::ios_base::beg);

    return bytesAvailable;
}

std::istream& skipMagic(std::istream& blobStream) {
    if (!blobStream.good()) {
        throw std::runtime_error("Can't execute skipMagic: an I/O error is occured");
    }

    using ExportMagic = std::array<char, 4>;
    constexpr static const ExportMagic exportMagic = {{0x1, 0xE, 0xE, 0x1}};
    ExportMagic magic = {};

    blobStream.seekg(0, blobStream.beg);
    blobStream.read(magic.data(), magic.size());
    auto exportedWithName = (exportMagic == magic);
    if (exportedWithName) {
        std::string tmp;
        std::getline(blobStream, tmp);
    } else {
        blobStream.seekg(0, blobStream.beg);
    }

    return blobStream;
}
}  // namespace vpux

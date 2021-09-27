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

#include <elf/utils/utils.hpp>

using namespace elf;

size_t utils::getStreamSize(std::istream& strm) {
    const auto streamStart = strm.tellg();
    strm.seekg(0, std::ios_base::end);
    const auto streamEnd = strm.tellg();
    const auto bytesAvailable = streamEnd - streamStart;
    strm.seekg(streamStart, std::ios_base::beg);

    return bytesAvailable;
}

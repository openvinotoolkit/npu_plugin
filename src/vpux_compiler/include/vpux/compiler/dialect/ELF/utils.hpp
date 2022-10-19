//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vector>
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace ELF {

std::pair<uint8_t*, size_t> getDataAndSizeOfElfSection(const std::vector<char>& elfBlob,
                                                       const std::vector<std::string> possibleSecNames);

}  // namespace ELF
}  // namespace vpux

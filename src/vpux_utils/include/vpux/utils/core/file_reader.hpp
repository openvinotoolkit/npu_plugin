//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <fstream>
#include <string>

namespace vpux {

size_t getFileSize(std::istream& strm);
std::istream& skipMagic(std::istream& blobStream);

}  // namespace vpux

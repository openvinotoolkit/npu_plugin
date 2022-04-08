//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#if defined(__arm__) || defined(__aarch64__)
#include <NnCorePlg.h>
#include <string>

namespace vpu {
namespace utils {

off_t bitsPerPixel(const DataType dType);
off_t getTotalSize(const TensorRefNDData& tensorRef);
std::string serializeShape(const TensorRefNDData& tensorRef);
std::string serializeStrides(const TensorRefNDData& tensorRef);
std::string serializeDType(const TensorRefNDData& tensorRef);
std::string serializeOrder(const TensorRefNDData& tensorRef);

}  // namespace utils
}  // namespace vpu
#endif

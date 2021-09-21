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

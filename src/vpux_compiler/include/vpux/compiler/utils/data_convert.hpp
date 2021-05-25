//
// Copyright 2020 Intel Corporation.
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

#include "vpux/utils/IE/float16.hpp"

#include <mlir/IR/Types.h>

namespace vpux {

template <typename T>
T convertData(const char* data, mlir::Type baseType);

extern template uint8_t convertData<uint8_t>(const char* data, mlir::Type baseType);
extern template uint16_t convertData<uint16_t>(const char* data, mlir::Type baseType);
extern template uint32_t convertData<uint32_t>(const char* data, mlir::Type baseType);
extern template uint64_t convertData<uint64_t>(const char* data, mlir::Type baseType);

extern template int8_t convertData<int8_t>(const char* data, mlir::Type baseType);
extern template int16_t convertData<int16_t>(const char* data, mlir::Type baseType);
extern template int32_t convertData<int32_t>(const char* data, mlir::Type baseType);
extern template int64_t convertData<int64_t>(const char* data, mlir::Type baseType);

extern template float convertData<float>(const char* data, mlir::Type baseType);
extern template double convertData<double>(const char* data, mlir::Type baseType);
extern template float16 convertData<float16>(const char* data, mlir::Type baseType);
extern template bfloat16 convertData<bfloat16>(const char* data, mlir::Type baseType);

}  // namespace vpux

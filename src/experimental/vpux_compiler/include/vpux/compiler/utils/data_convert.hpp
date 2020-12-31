//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <mlir/IR/Types.h>

#include <ngraph/type/bfloat16.hpp>
#include <ngraph/type/float16.hpp>

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
extern template ngraph::float16 convertData<ngraph::float16>(const char* data, mlir::Type baseType);
extern template ngraph::bfloat16 convertData<ngraph::bfloat16>(const char* data, mlir::Type baseType);

}  // namespace vpux

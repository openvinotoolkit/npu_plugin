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

#include "vpux/compiler/utils/data_convert.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

namespace {

template <typename Out>
struct CvtHelper final {
    template <typename In>
    static Out cvt(In val) {
        return checked_cast<Out>(val);
    }
};

template <>
struct CvtHelper<float16> final {
    template <typename In>
    static float16 cvt(In val) {
        return float16(checked_cast<float>(val));
    }
};

template <>
struct CvtHelper<bfloat16> final {
    template <typename In>
    static bfloat16 cvt(In val) {
        return bfloat16(checked_cast<float>(val));
    }
};

}  // namespace

template <typename T>
T vpux::convertData(const char* data, mlir::Type baseType) {
    if (baseType.isUnsignedInteger(8)) {
        return CvtHelper<T>::cvt(*reinterpret_cast<const uint8_t*>(data));
    } else if (baseType.isUnsignedInteger(16)) {
        return CvtHelper<T>::cvt(*reinterpret_cast<const uint16_t*>(data));
    } else if (baseType.isUnsignedInteger(32)) {
        return CvtHelper<T>::cvt(*reinterpret_cast<const uint32_t*>(data));
    } else if (baseType.isUnsignedInteger(64)) {
        return CvtHelper<T>::cvt(*reinterpret_cast<const uint64_t*>(data));
    } else if (baseType.isSignedInteger(8)) {
        return CvtHelper<T>::cvt(*reinterpret_cast<const int8_t*>(data));
    } else if (baseType.isSignedInteger(16)) {
        return CvtHelper<T>::cvt(*reinterpret_cast<const int16_t*>(data));
    } else if (baseType.isSignedInteger(32)) {
        return CvtHelper<T>::cvt(*reinterpret_cast<const int32_t*>(data));
    } else if (baseType.isSignedInteger(64)) {
        return CvtHelper<T>::cvt(*reinterpret_cast<const int64_t*>(data));
    } else if (baseType.isF32()) {
        return CvtHelper<T>::cvt(*reinterpret_cast<const float*>(data));
    } else if (baseType.isF64()) {
        return CvtHelper<T>::cvt(*reinterpret_cast<const double*>(data));
    } else if (baseType.isF16()) {
        const auto& temp = *reinterpret_cast<const float16*>(data);
        return CvtHelper<T>::cvt(static_cast<float>(temp));
    } else if (baseType.isBF16()) {
        const auto& temp = *reinterpret_cast<const bfloat16*>(data);
        return CvtHelper<T>::cvt(static_cast<float>(temp));
    } else {
        VPUX_THROW("Unsupported element type '{0}'", baseType);
    }
}

namespace vpux {

template uint8_t convertData<uint8_t>(const char* data, mlir::Type baseType);
template uint16_t convertData<uint16_t>(const char* data, mlir::Type baseType);
template uint32_t convertData<uint32_t>(const char* data, mlir::Type baseType);
template uint64_t convertData<uint64_t>(const char* data, mlir::Type baseType);

template int8_t convertData<int8_t>(const char* data, mlir::Type baseType);
template int16_t convertData<int16_t>(const char* data, mlir::Type baseType);
template int32_t convertData<int32_t>(const char* data, mlir::Type baseType);
template int64_t convertData<int64_t>(const char* data, mlir::Type baseType);

template float convertData<float>(const char* data, mlir::Type baseType);
template double convertData<double>(const char* data, mlir::Type baseType);
template float16 convertData<float16>(const char* data, mlir::Type baseType);
template bfloat16 convertData<bfloat16>(const char* data, mlir::Type baseType);

}  // namespace vpux

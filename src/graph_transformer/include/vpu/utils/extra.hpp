//
// Copyright (C) 2018-2019 Intel Corporation.
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

#include <details/ie_exception.hpp>
#include <ie_profiling.hpp>

namespace vpu {

//
// VPU_COMBINE
//

#define VPU_COMBINE_HELPER2(X, Y)  X##Y
#define VPU_COMBINE_HELPER3(X, Y, Z)  X##Y##Z

#define VPU_COMBINE(X, Y)   VPU_COMBINE_HELPER2(X, Y)
#define VPU_COMBINE3(X, Y, Z)   VPU_COMBINE_HELPER3(X, Y, Z)

//
// Exceptions
//

#define VPU_THROW_EXCEPTION \
    THROW_IE_EXCEPTION << "[VPU] "

#define VPU_THROW_UNLESS(EXPRESSION) \
    if (!(EXPRESSION)) VPU_THROW_EXCEPTION << "AssertionFailed: " << #EXPRESSION  // NOLINT

//
// Packed structure declaration
//

#ifdef _MSC_VER
#   define VPU_PACKED(body) __pragma(pack(push, 1)) struct body __pragma(pack(pop))
#elif defined(__GNUC__)
#   define VPU_PACKED(body) struct __attribute__((packed)) body
#endif

//
// Profiling
//

#define VPU_PROFILE(NAME) IE_PROFILING_AUTO_SCOPE(VPU_ ## NAME)

}  // namespace vpu

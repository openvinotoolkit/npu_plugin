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

namespace vpux {

//
// VPUX_COMBINE
//

#define VPUX_COMBINE_HELPER2(_X_, _Y_) _X_##_Y_
#define VPUX_COMBINE(_X_, _Y_) VPUX_COMBINE_HELPER2(_X_, _Y_)

#define VPUX_COMBINE_HELPER3(_X_, _Y_, _Z_) _X_##_Y_##_Z_
#define VPUX_COMBINE3(_X, _Y_, _Z_) VPUX_COMBINE_HELPER3(_X_, _Y_, _Z_)

//
// VPUX_UNIQUE_NAME
//

#ifdef __COUNTER__
#define VPUX_UNIQUE_NAME(_BaseName_) VPUX_COMBINE(_BaseName_, __COUNTER__)
#else
#define VPUX_UNIQUE_NAME(_BaseName_) VPUX_COMBINE(_BaseName_, __LINE__)
#endif

//
// VPUX_UNUSED
//

#define VPUX_UNUSED(_var_) (void)(_var_)

//
// VPUX_PACKED
//

#ifdef _MSC_VER
#define VPUX_PACKED(body) __pragma(pack(push, 1)) struct body __pragma(pack(pop))
#elif defined(__GNUC__)
#define VPUX_PACKED(body) struct __attribute__((packed)) body
#endif

//
// VPUX_EXPAND
//

#define VPUX_EXPAND(x) x

}  // namespace vpux

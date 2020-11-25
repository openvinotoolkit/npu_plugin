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

#if defined(__GNUC__)
#define VPUX_PACKED __attribute__((packed))
#else
#define VPUX_PACKED
#endif

//
// VPUX_EXPAND
//

#define VPUX_EXPAND(x) x

}  // namespace vpux

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

#include "vpux/utils/core/mem_size.hpp"

using namespace vpux;

StringLiteral vpux::stringifyEnum(MemType val) {
    switch (val) {
    case MemType::Byte:
        return "Byte";
    case MemType::KB:
        return "KB";
    case MemType::MB:
        return "MB";
    case MemType::GB:
        return "GB";
    default:
        return "<UNKNOWN>";
    }
}

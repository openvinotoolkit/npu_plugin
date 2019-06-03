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

#include <vpu/utils/io.hpp>

#include <iostream>

#include <vpu/utils/any.hpp>
#include <vpu/utils/attributes_map.hpp>
#include <vpu/utils/extra.hpp>

namespace vpu {

void printTo(std::ostream& os, const Any& any) noexcept {
    any.printImpl(os);
}

void printTo(std::ostream& os, const AttributesMap& attrs) noexcept {
    attrs.printImpl(os);
}

void formatPrint(std::ostream& os, const char* str) noexcept {
    try {
        while (*str) {
            if (*str == '%') {
                if (*(str + 1) == '%') {
                    ++str;
                } else {
                    throw std::invalid_argument("[VPU] Invalid format string : missing arguments");
                }
            }

            os << *str++;
        }
    } catch (std::invalid_argument e) {
        std::cerr << e.what() << '\n';
        std::abort();
    } catch (...) {
        std::cerr << "[VPU] Unknown error in formatPrint\n";
        std::abort();
    }
}

}  // namespace vpu

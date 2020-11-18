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

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <gtest/gtest.h>

enum class TestEnum { A, B };

namespace vpux {

template <>
struct EnumTraits<TestEnum> {
    static StringRef getEnumValueName(TestEnum val) {
#define CASE(_val_)                                                            \
    case TestEnum::_val_:                                                      \
        return #_val_

        switch (val) {
            CASE(A);
            CASE(B);
        default:
            return "<UNKNOWN>";
        }

#undef CASE
    }
};

}  // namespace vpux

TEST(PlainEnumTest, Format) {
    EXPECT_EQ(llvm::formatv("{0}", TestEnum::A).str(), "A");
    EXPECT_EQ(llvm::formatv("{0}", TestEnum::B).str(), "B");
}

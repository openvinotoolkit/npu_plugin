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

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <gtest/gtest.h>

enum class TestEnum { A, B };

vpux::StringLiteral stringifyEnum(TestEnum val) {
#define CASE(_val_)       \
    case TestEnum::_val_: \
        return #_val_

    switch (val) {
        CASE(A);
        CASE(B);
    default:
        return "<UNKNOWN>";
    }

#undef CASE
}

TEST(MLIR_PlainEnumTest, Format) {
    EXPECT_EQ(llvm::formatv("{0}", TestEnum::A).str(), "A");
    EXPECT_EQ(llvm::formatv("{0}", TestEnum::B).str(), "B");
}

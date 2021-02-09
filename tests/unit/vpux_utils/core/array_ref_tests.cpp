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

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace vpux;

template <class Range>
void cmpArrayRefWithRange(const Range& range, ArrayRef<typename Range::value_type> arr) {
    size_t ind = 0;
    for (const auto& v : range) {
        ASSERT_LT(ind, arr.size());
        EXPECT_EQ(&v, &arr[ind]) << ind;
        ++ind;
    }
}

TEST(MLIR_ArrayRefTest, Empty) {
    ArrayRef<int> empty_arr1;
    EXPECT_TRUE(empty_arr1.empty());
}

TEST(MLIR_ArrayRefTest, FromPlainArray) {
    int elems[] = {1, 2, 3, 4, 5};

    ArrayRef<int> arr1(elems);
    ASSERT_EQ(5, arr1.size());
    cmpArrayRefWithRange(make_range(elems), arr1);

    ArrayRef<int> arr2(elems, 5);
    ASSERT_EQ(5, arr2.size());
    cmpArrayRefWithRange(make_range(elems), arr2);

    ArrayRef<int> arr3(elems, elems + 5);
    ASSERT_EQ(5, arr3.size());
    cmpArrayRefWithRange(make_range(elems), arr3);
}

template <class Cont>
class ArrayRefContTest : public testing::Test {
public:
    using cont_type = Cont;
    using value_type = typename Cont::value_type;
    using array_ref_type = ArrayRef<value_type>;

    void testFromContainer() {
        cont_type cont = {1, 2, 3, 4, 5};
        array_ref_type arr = cont;

        ASSERT_EQ(cont.size(), arr.size());
        cmpArrayRefWithRange(cont, arr);
    }
};

using ArrayRefContTestTypes = testing::Types<std::vector<int>, std::array<int, 5>, SmallVector<int, 10>,
                                             SmallVector<int, 2>, std::initializer_list<int>>;
TYPED_TEST_CASE(ArrayRefContTest, ArrayRefContTestTypes);

TYPED_TEST(ArrayRefContTest, FromContainer) {
    this->testFromContainer();
}
